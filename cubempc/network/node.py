from __future__ import annotations
import asyncio
import logging
from collections import deque
from collections.abc import Callable
from typing import Any
from cubempc.config import ExperimentConfig, LayerState, MPCConfig
from cubempc.host_layout import HostLayout
from cubempc.messages import TRANSPORT_JSON, TRANSPORT_OBJECT, Message, decode_message, encode_message, frame_tcp, unframe_tcp_length
from cubempc.metrics import Metrics
logger = logging.getLogger(__name__)
FilterFn = Callable[[Message], bool]
InboxKey = tuple[int, str, str, str, int]

def message_inbox_key(msg: Message, receiver_rank: int) -> InboxKey:
    return (msg.dst_layer, msg.protocol, msg.instance_id, msg.msg_type, receiver_rank)
SAME_LAYER_PROTOCOLS: frozenset[str] = frozenset({'ping', 'control', 'layer_demo', 'vss', 'mul'})

def logical_party(rank: int, layer: int) -> str:
    return f'P_{rank}^{layer}'

def _resolve_config(config: MPCConfig | ExperimentConfig) -> tuple[MPCConfig, HostLayout]:
    if isinstance(config, MPCConfig):
        return config, config.host_layout()
    mpc = config.to_mpc_config()
    return mpc, mpc.host_layout()

async def _read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    data = await reader.readexactly(n)
    if len(data) < n:
        raise asyncio.IncompleteReadError(data, n)
    return data

async def _read_framed_body(reader: asyncio.StreamReader) -> bytes:
    prefix = await _read_exact(reader, 4)
    length = unframe_tcp_length(prefix)
    return await _read_exact(reader, length)

class NodeProcess:

    def __init__(self, rank: int, n: int, base_port: int, config: MPCConfig | ExperimentConfig) -> None:
        mpc, host_layout = _resolve_config(config)
        if rank < 0 or rank >= n:
            raise ValueError(f'rank {rank} out of range [0, {n})')
        if n != mpc.n:
            raise ValueError(f'n={n} does not match config.n={mpc.n}')
        self.rank = rank
        self.n = n
        self.base_port = base_port
        self.mpc_config = mpc
        self.config = config
        self.host_layout = host_layout
        self.host = host_layout.host_for_rank(rank)
        self.listen_host = '127.0.0.1' if host_layout.is_local_only() else '0.0.0.0'
        self.port = base_port + rank
        self.pending_messages: list[Message] = []
        self.inbox: dict[InboxKey, deque[Message]] = {}
        self._message_condition = asyncio.Condition()
        self.metrics = Metrics()
        self.layer_states: dict[int, LayerState] = {}
        self._server: asyncio.AbstractServer | None = None
        self.peer_writers: dict[int, asyncio.StreamWriter] = {}
        self._peer_send_locks: dict[int, asyncio.Lock] = {}
        self.transport_payload_mode: str = TRANSPORT_OBJECT

    def logical_id(self, layer: int) -> str:
        return logical_party(self.rank, layer)

    def get_state(self, layer: int) -> LayerState:
        if layer not in self.layer_states:
            self.layer_states[layer] = LayerState(layer=layer)
        return self.layer_states[layer]

    def clear_state(self, layer: int) -> None:
        self.layer_states.pop(layer, None)

    def clear_before(self, layer: int) -> None:
        for key in [k for k in self.layer_states if k < layer]:
            del self.layer_states[key]

    def clear_before_layer(self, layer_id: int) -> None:
        self.clear_before(layer_id)
        for key in [k for k in self.inbox if k[0] < layer_id]:
            del self.inbox[key]
        self.pending_messages = [m for m in self.pending_messages if m.dst_layer >= layer_id]

    def clear_instance_state(self, protocol: str, instance_id: str) -> None:
        for key in [k for k in self.inbox if k[1] == protocol and k[2] == instance_id]:
            del self.inbox[key]
        self.pending_messages = [m for m in self.pending_messages if not (m.protocol == protocol and m.instance_id == instance_id)]
        for state in self.layer_states.values():
            inbound = state.values.get('inbound')
            if isinstance(inbound, list):
                state.values['inbound'] = [m for m in inbound if not (m.protocol == protocol and m.instance_id == instance_id)]
            for key in list(state.protocol_states):
                if key == instance_id:
                    del state.protocol_states[key]
                elif isinstance(key, tuple) and key and (key[0] == instance_id):
                    del state.protocol_states[key]
            for key in list(state.values):
                if key == 'inbound':
                    continue
                if key == instance_id:
                    del state.values[key]
                elif isinstance(key, tuple) and key and (key[0] == instance_id):
                    del state.values[key]

    def _receiver_rank(self, msg: Message) -> int:
        if msg.dst_rank is not None:
            return msg.dst_rank
        return self.rank

    def _store_inbound(self, msg: Message) -> None:
        state = self.get_state(msg.dst_layer)
        inbound: list[Message] = state.values.setdefault('inbound', [])
        inbound.append(msg)

    def _enqueue_inbox(self, msg: Message) -> None:
        key = message_inbox_key(msg, self._receiver_rank(msg))
        self.inbox.setdefault(key, deque()).append(msg)

    def _remove_pending_message(self, msg: Message) -> None:
        for idx, pending in enumerate(self.pending_messages):
            if pending is msg:
                del self.pending_messages[idx]
                return

    def _remove_from_inbox(self, msg: Message) -> None:
        key = message_inbox_key(msg, self._receiver_rank(msg))
        bucket = self.inbox.get(key)
        if bucket is None:
            return
        filtered = deque((m for m in bucket if m is not msg))
        if filtered:
            self.inbox[key] = filtered
        else:
            self.inbox.pop(key, None)

    def _collect_indexed(self, inbox_key: InboxKey, filter_fn: FilterFn, need: int) -> list[Message]:
        if need <= 0:
            return []
        bucket = self.inbox.get(inbox_key)
        if not bucket:
            return []
        matched: list[Message] = []
        deferred: deque[Message] = deque()
        while bucket and len(matched) < need:
            msg = bucket.popleft()
            if filter_fn(msg):
                matched.append(msg)
                self._remove_pending_message(msg)
            else:
                deferred.append(msg)
        while deferred:
            bucket.appendleft(deferred.pop())
        if not bucket:
            self.inbox.pop(inbox_key, None)
        return matched

    def _scan_pending(self, filter_fn: FilterFn, need: int) -> list[Message]:
        if need <= 0:
            return []
        matched: list[Message] = []
        kept: list[Message] = []
        for msg in self.pending_messages:
            if len(matched) < need and filter_fn(msg):
                matched.append(msg)
                self._remove_from_inbox(msg)
            else:
                kept.append(msg)
        self.pending_messages = kept
        return matched

    async def _queue_inbound(self, msg: Message) -> None:
        self._store_inbound(msg)
        async with self._message_condition:
            self.pending_messages.append(msg)
            self._enqueue_inbox(msg)
            self._message_condition.notify_all()

    def peer_port(self, dst_rank: int) -> int:
        if dst_rank < 0 or dst_rank >= self.n:
            raise ValueError(f'dst_rank {dst_rank} out of range [0, {self.n})')
        return self.base_port + dst_rank

    def _peer_send_lock(self, dst_rank: int) -> asyncio.Lock:
        if dst_rank not in self._peer_send_locks:
            self._peer_send_locks[dst_rank] = asyncio.Lock()
        return self._peer_send_locks[dst_rank]

    def _peer_ranks(self) -> list[int]:
        return [r for r in range(self.n) if r != self.rank]

    async def _close_peer_writer(self, dst_rank: int) -> None:
        writer = self.peer_writers.pop(dst_rank, None)
        if writer is None:
            return
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def close_peer_connections(self) -> None:
        for dst_rank in list(self.peer_writers):
            await self._close_peer_writer(dst_rank)

    async def _connect_peer(self, dst_rank: int) -> asyncio.StreamWriter:
        if dst_rank == self.rank:
            raise ValueError('cannot open persistent connection to self')
        await self._close_peer_writer(dst_rank)
        port = self.peer_port(dst_rank)
        peer_host = self.host_layout.host_for_rank(dst_rank)
        _reader, writer = await asyncio.open_connection(peer_host, port)
        self.peer_writers[dst_rank] = writer
        logger.debug('rank %s connected to peer %s on %s:%s', self.rank, dst_rank, peer_host, port)
        return writer

    async def connect_peers(self, *, timeout: float=30.0) -> None:
        pending = set(self._peer_ranks())
        if not pending:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_error: Exception | None = None
        while pending and loop.time() < deadline:
            for dst_rank in list(pending):
                try:
                    await self._connect_peer(dst_rank)
                    pending.discard(dst_rank)
                except OSError as exc:
                    last_error = exc
            if pending:
                await asyncio.sleep(0.05)
        if pending:
            msg = f'rank {self.rank}: failed to connect to peer ranks {sorted(pending)} within {timeout}s'
            if last_error is not None:
                raise ConnectionError(msg) from last_error
            raise ConnectionError(msg)

    async def _deliver_local(self, msg: Message) -> None:
        await self._queue_inbound(msg)

    async def _get_peer_writer(self, dst_rank: int) -> asyncio.StreamWriter:
        if dst_rank == self.rank:
            raise ValueError('use _deliver_local for self rank')
        writer = self.peer_writers.get(dst_rank)
        if writer is not None and (not writer.is_closing()):
            return writer
        return await self._connect_peer(dst_rank)

    async def _write_framed(self, dst_rank: int, framed: bytes) -> None:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                writer = await self._get_peer_writer(dst_rank)
                async with self._peer_send_lock(dst_rank):
                    writer.write(framed)
                    await writer.drain()
                return
            except (asyncio.IncompleteReadError, ConnectionError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                last_exc = exc
                await self._close_peer_writer(dst_rank)
                if attempt == 0:
                    logger.debug('rank %s send to %s failed (%s), reconnecting', self.rank, dst_rank, exc)
                    continue
                raise
        if last_exc is not None:
            raise last_exc

    async def start_server(self, *, bind_retries: int=30, bind_retry_delay: float=0.2) -> asyncio.AbstractServer:
        last_err: OSError | None = None
        for attempt in range(bind_retries):
            try:
                self._server = await asyncio.start_server(self._handle_client, self.listen_host, self.port, reuse_address=True)
                logger.info('physical rank %s listening on %s:%s (peer host %s)', self.rank, self.listen_host, self.port, self.host)
                return self._server
            except OSError as exc:
                last_err = exc
                if exc.errno != 98 or attempt + 1 >= bind_retries:
                    raise
                await asyncio.sleep(bind_retry_delay * (attempt + 1))
        if last_err is not None:
            raise last_err
        raise RuntimeError('start_server failed without error')

    async def stop_server(self) -> None:
        await self.close_peer_connections()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                body = await _read_framed_body(reader)
                msg = decode_message(body)
                if msg.channel == 'broadcast' and msg.dst_rank is None:
                    msg.dst_rank = self.rank
                await self._queue_inbound(msg)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def send_layer_msg(self, src_layer: int, dst_layer: int, dst_rank: int, protocol: str, instance_id: str, msg_type: str, payload: object, *, channel: str='p2p') -> None:
        if dst_layer == src_layer + 1:
            pass
        elif dst_layer == src_layer and protocol in SAME_LAYER_PROTOCOLS:
            pass
        else:
            raise ValueError(f'invalid layer route {src_layer}->{dst_layer} for protocol {protocol}')
        if channel == 'broadcast':
            msg = Message(src_rank=self.rank, dst_rank=None, src_layer=src_layer, dst_layer=dst_layer, protocol=protocol, instance_id=instance_id, msg_type=msg_type, channel='broadcast', payload=payload)
            await self.broadcast(msg)
            return
        msg = Message(src_rank=self.rank, dst_rank=dst_rank, src_layer=src_layer, dst_layer=dst_layer, protocol=protocol, instance_id=instance_id, msg_type=msg_type, channel='p2p', payload=payload)
        if dst_layer == src_layer + 1:
            msg.validate_routing()
        await self.send_p2p(dst_rank, msg)

    async def send_p2p(self, dst_rank: int, msg: Message) -> None:
        msg.channel = 'p2p'
        msg.dst_rank = dst_rank
        if dst_rank == self.rank:
            await self._deliver_local(msg)
            self.metrics.record_p2p()
            return
        encoded = encode_message(msg, transport=self.transport_payload_mode)
        framed = frame_tcp(encoded)
        await self._write_framed(dst_rank, framed)
        self.metrics.record_p2p(len(encoded))

    async def broadcast(self, msg: Message) -> None:
        msg.channel = 'broadcast'
        wire_msg = Message(src_rank=msg.src_rank, dst_rank=None, src_layer=msg.src_layer, dst_layer=msg.dst_layer, protocol=msg.protocol, instance_id=msg.instance_id, msg_type=msg.msg_type, channel='broadcast', payload=msg.payload)
        encoded = encode_message(wire_msg, transport=self.transport_payload_mode)
        framed = frame_tcp(encoded)
        local_msg = Message(src_rank=msg.src_rank, dst_rank=self.rank, src_layer=msg.src_layer, dst_layer=msg.dst_layer, protocol=msg.protocol, instance_id=msg.instance_id, msg_type=msg.msg_type, channel='broadcast', payload=msg.payload)
        remote_peers = self._peer_ranks()
        tasks: list[asyncio.Task[None]] = [asyncio.create_task(self._deliver_local(local_msg))]
        tasks.extend((asyncio.create_task(self._write_framed(dst, framed)) for dst in remote_peers))
        await asyncio.gather(*tasks)
        physical_total = len(framed) * len(remote_peers)
        self.metrics.record_broadcast(len(encoded), physical_size=physical_total)

    async def recv_until(self, filter_fn: FilterFn, expected_count: int, timeout: float, *, inbox_key: InboxKey | None=None) -> list[Message]:
        matched: list[Message] = []
        deadline = asyncio.get_running_loop().time() + timeout
        async with self._message_condition:
            while len(matched) < expected_count:
                need = expected_count - len(matched)
                if inbox_key is not None:
                    matched.extend(self._collect_indexed(inbox_key, filter_fn, need))
                    need = expected_count - len(matched)
                if need > 0:
                    matched.extend(self._scan_pending(filter_fn, need))
                if len(matched) >= expected_count:
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError(f'rank {self.rank}: got {len(matched)}/{expected_count} messages')
                await asyncio.wait_for(self._message_condition.wait(), timeout=remaining)
        return matched