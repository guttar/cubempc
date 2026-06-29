from __future__ import annotations
import asyncio
import hashlib
from dataclasses import dataclass
from random import Random
from typing import Any
from cubempc.circuits.circuit import Circuit, evaluate_plain, validate_circuit
from cubempc.field import add, mod, mul, rand_field
from cubempc.protocols.multiplication import resharing_prep_keys, run_mul
from cubempc.protocols.resharing import run_resharing
from cubempc.protocols.rg import run_rg
from cubempc.protocols.vss import run_vss
from cubempc.shamir import robust_reconstruct, shamir_share
Node = Any
VSS_START_LAYER = 0
VSS_OUTPUT_LAYER = 3

def party_id(rank: int) -> int:
    return rank + 1

@dataclass
class WireState:
    layer: int
    share_key: Any

def vss_share_key(instance_id: str, wire: str) -> tuple[str, str]:
    return (f'{instance_id}:input:{wire}', 'vss_share')

def wire_value_key(instance_id: str, wire: str) -> tuple[str, str]:
    return (instance_id, 'wire', wire)

def generate_random_sharings_to_layer(node: Node, layer: int, key_prefix: str, count: int, rng: Random) -> list[Any]:
    n = node.n
    t = node.mpc_config.t
    state = node.get_state(layer)
    keys: list[Any] = []
    rank_party = party_id(node.rank)
    for i in range(count):
        key = (key_prefix, 'rand', i)
        keys.append(key)
        secret = rand_field(rng)
        shares = shamir_share(secret, n, t, rng=rng)
        state.values[key] = shares[rank_party]
    return keys

class MPCPreprocessingPool:

    def __init__(self, node: Node, mpc_instance_id: str, rng: Random, *, mode: str, timing: dict[str, Any] | None=None) -> None:
        if mode not in {'local', 'rg'}:
            raise ValueError(f'unsupported randomness_mode {mode!r}')
        self.node = node
        self.mpc_instance_id = mpc_instance_id
        self.rng = rng
        self.mode = mode
        self.timing = timing
        self._buffers: dict[int, list[int]] = {}
        self._rg_counter = 0
        self._local_counter = 0
        self._consume_lock = asyncio.Lock()

    async def consume_to_keys(self, layer: int, key_prefix: str, count: int, *, keys: list[Any] | None=None, circuit_depth: int | None=None) -> list[Any]:
        if count < 0:
            raise ValueError(f'count must be non-negative, got {count}')
        if keys is None:
            out_keys = [(key_prefix, 'rand', i) for i in range(count)]
        else:
            if len(keys) != count:
                raise ValueError(f'expected {count} keys, got {len(keys)}')
            out_keys = keys
        async with self._consume_lock:
            values = await self._consume_values(layer, count, circuit_depth=circuit_depth)
            state = self.node.get_state(layer)
            for key, value in zip(out_keys, values, strict=True):
                state.values[key] = mod(value)
        return out_keys

    async def _consume_values(self, layer: int, count: int, *, circuit_depth: int | None=None) -> list[int]:
        if count == 0:
            return []
        if self.mode == 'local':
            n = self.node.n
            t = self.node.mpc_config.t
            rank_party = party_id(self.node.rank)
            values: list[int] = []
            for _ in range(count):
                seed = int.from_bytes(hashlib.sha256(f'{self.mpc_instance_id}:local-rand:{layer}:{self._local_counter}'.encode()).digest()[:8], 'big')
                self._local_counter += 1
                local_rng = Random(seed)
                secret = rand_field(local_rng)
                shares = shamir_share(secret, n, t, rng=local_rng)
                values.append(shares[rank_party])
            return values
        if layer < 3:
            raise ValueError(f'Π_RG output for layer {layer} needs start_layer >= 0')
        buffer = self._buffers.setdefault(layer, [])
        while len(buffer) < count:
            await self._refill_rg(layer, circuit_depth=circuit_depth)
        values = buffer[:count]
        del buffer[:count]
        return values

    async def _refill_rg(self, layer: int, *, circuit_depth: int | None=None) -> None:
        q = self.node.n - self.node.mpc_config.t
        rg_inst = f'{self.mpc_instance_id}:rg:L{layer}:{self._rg_counter}'
        self._rg_counter += 1
        rg_timing: dict[str, Any] = {}
        shares = await run_rg(self.node, layer - 3, q, rg_inst, rng=self.rng, timing=rg_timing)
        self._buffers.setdefault(layer, []).extend((mod(v) for v in shares))
        if self.timing is not None:
            self.timing.setdefault('rg_parallel_ms', []).append(float(rg_timing.get('parallel_equiv_ms', 0.0)))
            self.timing.setdefault('rg_stages', []).append({'depth': circuit_depth, 'layer': layer, 'ms': float(rg_timing.get('parallel_equiv_ms', 0.0))})
            self.timing.setdefault('rg_sequential_impl_ms', []).append(float(rg_timing.get('sequential_impl_ms', 0.0)))

def populate_resharing_prep_keys(node: Node, layer: int, instance_id: str, rng: Random, pool: MPCPreprocessingPool | None=None) -> list[Any]:
    n = node.n
    t = node.mpc_config.t
    state = node.get_state(layer)
    keys = resharing_prep_keys(instance_id, t)
    if pool is not None:
        raise RuntimeError('populate_resharing_prep_keys with pool must be awaited')
    rank_party = party_id(node.rank)
    for key in keys:
        secret = rand_field(rng)
        shares = shamir_share(secret, n, t, rng=rng)
        state.values[key] = shares[rank_party]
    return keys

async def populate_resharing_prep_keys_async(node: Node, layer: int, instance_id: str, rng: Random, pool: MPCPreprocessingPool | None, *, circuit_depth: int | None=None) -> list[Any]:
    t = node.mpc_config.t
    keys = resharing_prep_keys(instance_id, t)
    if pool is None:
        return populate_resharing_prep_keys(node, layer, instance_id, rng)
    return await pool.consume_to_keys(layer, instance_id, t, keys=keys, circuit_depth=circuit_depth)

def read_share(node: Node, ws: WireState) -> int:
    return mod(node.get_state(ws.layer).values[ws.share_key])

def write_share(node: Node, ws: WireState, value: int) -> None:
    node.get_state(ws.layer).values[ws.share_key] = mod(value)

async def reshare_wire_to_next_layer(node: Node, ws: WireState, wire_name: str, mpc_instance_id: str, rng: Random, *, pool: MPCPreprocessingPool | None=None, timing: dict[str, Any] | None=None, circuit_depth: int | None=None, recv_timeout: float=60.0) -> WireState:
    src = ws.layer
    rs_inst = f'{mpc_instance_id}:align:{wire_name}:L{src}'
    prep_keys = await populate_resharing_prep_keys_async(node, src, rs_inst, rng, pool, circuit_depth=circuit_depth)
    out_key = wire_value_key(mpc_instance_id, f'{wire_name}@L{src + 1}')
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await run_resharing(node=node, layer=src, input_share_key=ws.share_key, random_share_keys=prep_keys, output_share_key=out_key, instance_id=rs_inst, recv_timeout=recv_timeout)
    if timing is not None:
        elapsed_ms = (loop.time() - t0) * 1000.0
        timing.setdefault('rs_ms', []).append(elapsed_ms)
        timing.setdefault('rs_stages', []).append({'depth': circuit_depth, 'layer': src, 'ms': elapsed_ms})
    return WireState(layer=src + 1, share_key=out_key)

async def align_wire_to_layer(node: Node, wires: dict[str, WireState], wire_name: str, target_layer: int, mpc_instance_id: str, rng: Random, *, pool: MPCPreprocessingPool | None=None, timing: dict[str, Any] | None=None, circuit_depth: int | None=None, recv_timeout: float=60.0) -> WireState:
    ws = wires[wire_name]
    while ws.layer < target_layer:
        ws = await reshare_wire_to_next_layer(node, ws, wire_name, mpc_instance_id, rng, pool=pool, timing=timing, circuit_depth=circuit_depth, recv_timeout=recv_timeout)
    wires[wire_name] = ws
    return ws

def prepare_mul_randomness(node: Node, layer: int, mul_instance_id: str, rng: Random) -> list[Any]:
    t = node.mpc_config.t
    count = 3 * t + 1
    random_keys = generate_random_sharings_to_layer(node, layer, mul_instance_id, count, rng)
    populate_resharing_prep_keys(node, layer, mul_instance_id, rng)
    return random_keys

async def prepare_mul_randomness_async(node: Node, layer: int, mul_instance_id: str, rng: Random, pool: MPCPreprocessingPool | None, *, circuit_depth: int | None=None) -> list[Any]:
    if pool is None:
        return prepare_mul_randomness(node, layer, mul_instance_id, rng)
    t = node.mpc_config.t
    random_keys = await pool.consume_to_keys(layer, mul_instance_id, 3 * t + 1, circuit_depth=circuit_depth)
    await populate_resharing_prep_keys_async(node, layer, mul_instance_id, rng, pool, circuit_depth=circuit_depth)
    return random_keys

async def run_cubempc(node: Node, circuit: Circuit, input_values: dict[str, int], input_dealers: dict[str, int], instance_id: str, rng: Random | None=None, *, randomness_mode: str='local', timing: dict[str, Any] | None=None, recv_timeout: float=120.0) -> int:
    validate_circuit(circuit)
    gen = rng if rng is not None else Random(0)
    if timing is not None:
        timing.setdefault('input_vss_ms', [])
        timing.setdefault('input_vss_detail_ms', [])
        timing.setdefault('rs_ms', [])
        timing.setdefault('mul_ms', [])
        timing.setdefault('rg_parallel_ms', [])
        timing.setdefault('rg_sequential_impl_ms', [])
        timing.setdefault('rs_stages', [])
        timing.setdefault('mul_stages', [])
        timing.setdefault('rg_stages', [])
    pool = MPCPreprocessingPool(node, instance_id, gen, mode=randomness_mode, timing=timing)
    wires: dict[str, WireState] = {}

    async def _share_input_wire(wire: str) -> tuple[str, float]:
        if wire not in input_dealers:
            raise KeyError(f'input wire {wire!r} missing from input_dealers')
        dealer_rank = input_dealers[wire]
        secret = mod(input_values[wire]) if node.rank == dealer_rank else 0
        vss_id = f'{instance_id}:input:{wire}'
        vss_seed = int.from_bytes(hashlib.sha256(f'{vss_id}:rank:{node.rank}'.encode()).digest()[:8], 'big')
        vss_rng = Random(vss_seed)
        t0 = asyncio.get_running_loop().time()
        await run_vss(node=node, start_layer=VSS_START_LAYER, dealer_rank=dealer_rank, secret=secret, instance_id=vss_id, rng=vss_rng)
        return (wire, (asyncio.get_running_loop().time() - t0) * 1000.0)
    input_t0 = asyncio.get_running_loop().time()
    input_results = await asyncio.gather(*(_share_input_wire(wire) for wire in circuit.inputs))
    input_wall_ms = (asyncio.get_running_loop().time() - input_t0) * 1000.0
    if timing is not None:
        timing.setdefault('input_vss_ms', []).append(input_wall_ms)
        timing.setdefault('input_vss_detail_ms', []).extend((elapsed_ms for _wire, elapsed_ms in input_results))
    for wire, _elapsed_ms in input_results:
        wires[wire] = WireState(layer=VSS_OUTPUT_LAYER, share_key=vss_share_key(instance_id, wire))

    async def _run_mul_gate(gate: Any, target: int) -> tuple[str, WireState]:
        assert gate.in1 is not None and gate.in2 is not None
        mul_inst = f'{instance_id}:mul:{gate.gid}'
        mul_seed = int.from_bytes(hashlib.sha256(f'{instance_id}:{gate.gid}:mul-rand'.encode()).digest()[:8], 'big')
        mul_rng = Random(mul_seed)
        random_keys = await prepare_mul_randomness_async(node, target, mul_inst, mul_rng, pool, circuit_depth=gate.depth)
        out_key = wire_value_key(instance_id, gate.gid)
        x_key = wires[gate.in1].share_key
        y_key = wires[gate.in2].share_key
        t0 = asyncio.get_running_loop().time()
        await run_mul(node, target, x_key, y_key, random_keys, out_key, mul_inst, recv_timeout=recv_timeout)
        if timing is not None:
            elapsed_ms = (asyncio.get_running_loop().time() - t0) * 1000.0
            timing.setdefault('mul_ms', []).append(elapsed_ms)
            timing.setdefault('mul_stages', []).append({'depth': gate.depth, 'layer': target, 'ms': elapsed_ms})
        return (gate.gid, WireState(layer=target + 1, share_key=out_key))
    for gate in circuit.gates:
        if gate.op == 'input':
            continue
        if gate.op == 'cmul':
            assert gate.in1 is not None and gate.const is not None
            ws = wires[gate.in1]
            const = mod(gate.const)
            out_key = wire_value_key(instance_id, gate.gid)
            node.get_state(ws.layer).values[out_key] = mul(const, read_share(node, ws))
            wires[gate.gid] = WireState(layer=ws.layer, share_key=out_key)
            continue
        if gate.op not in {'add', 'mul'}:
            raise ValueError(f'unsupported gate op {gate.op!r}')
        assert gate.in1 is not None and gate.in2 is not None
        target = max(wires[gate.in1].layer, wires[gate.in2].layer)
        for wire in sorted((gate.in1, gate.in2)):
            if wires[wire].layer < target:
                await align_wire_to_layer(node, wires, wire, target, instance_id, gen, pool=pool, timing=timing, circuit_depth=gate.depth, recv_timeout=recv_timeout)
        if gate.op == 'add':
            x = read_share(node, wires[gate.in1])
            y = read_share(node, wires[gate.in2])
            out_key = wire_value_key(instance_id, gate.gid)
            node.get_state(target).values[out_key] = add(x, y)
            wires[gate.gid] = WireState(layer=target, share_key=out_key)
        else:
            gid, ws = await _run_mul_gate(gate, target)
            wires[gid] = ws
    out_ws = wires[circuit.output]
    return read_share(node, out_ws)

def reconstruct_output(shares_by_party: dict[int, int], t: int) -> int | None:
    return robust_reconstruct(shares_by_party, t, t)

@dataclass
class MPCNodeResult:
    rank: int
    pid: int
    output_share: int
    output_layer: int
    metrics: dict[str, int]
    timing: dict[str, Any]

def _find_output_layer(node: Node, instance_id: str, output_wire: str) -> int:
    key = wire_value_key(instance_id, output_wire)
    for layer_id, state in node.layer_states.items():
        if key in state.values:
            return layer_id
    vkey = vss_share_key(instance_id, output_wire)
    for layer_id, state in node.layer_states.items():
        if vkey in state.values:
            return layer_id
    return VSS_OUTPUT_LAYER

def _sum_global_stage_latencies(results: list[MPCNodeResult], key: str) -> float:
    values = [r.timing.get(key, []) for r in results]
    max_len = max((len(v) for v in values), default=0)
    total = 0.0
    for idx in range(max_len):
        total += max((float(v[idx]) for v in values if idx < len(v)))
    return total

def _stage_max_by_depth(results: list[MPCNodeResult], key: str) -> dict[int, float]:
    by_depth: dict[int, float] = {}
    for result in results:
        for item in result.timing.get(key, []):
            if not isinstance(item, dict):
                continue
            depth_raw = item.get('depth')
            if depth_raw is None:
                depth = -1
            else:
                depth = int(depth_raw)
            by_depth[depth] = max(by_depth.get(depth, 0.0), float(item.get('ms', 0.0)))
    return by_depth

def _aggregate_protocol_latency_ms(results: list[MPCNodeResult]) -> float:
    input_parallel = max((max((float(v) for v in r.timing.get('input_vss_ms', [])), default=0.0) for r in results), default=0.0)
    if any((r.timing.get('mul_stages') or r.timing.get('rg_stages') or r.timing.get('rs_stages') for r in results)):
        rg_by_depth = _stage_max_by_depth(results, 'rg_stages')
        rs_by_depth = _stage_max_by_depth(results, 'rs_stages')
        mul_by_depth = _stage_max_by_depth(results, 'mul_stages')
        depths = set(rg_by_depth) | set(rs_by_depth) | set(mul_by_depth)
        layered_latency = sum((rg_by_depth.get(depth, 0.0) + rs_by_depth.get(depth, 0.0) + mul_by_depth.get(depth, 0.0) for depth in depths))
        return input_parallel + layered_latency
    rg_parallel = _sum_global_stage_latencies(results, 'rg_parallel_ms')
    rs_latency = _sum_global_stage_latencies(results, 'rs_ms')
    mul_latency = _sum_global_stage_latencies(results, 'mul_ms')
    return input_parallel + rg_parallel + rs_latency + mul_latency

async def _run_mpc_node_async(rank: int, n: int, t: int | None, base_port: int, circuit: Circuit, input_values: dict[str, int], input_dealers: dict[str, int], instance_id: str, startup_delay: float, randomness_mode: str, hosts: tuple[str, ...] | None=None) -> MPCNodeResult:
    import hashlib
    import os
    from cubempc.config import MPCConfig
    from cubempc.network.coordinator import _metrics_to_dict
    from cubempc.network.node import NodeProcess
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    node = NodeProcess(rank, n, base_port, mpc)
    await node.start_server()
    await asyncio.sleep(startup_delay)
    await node.connect_peers()
    seed = int.from_bytes(hashlib.sha256(f'{instance_id}:mpc'.encode()).digest()[:8], 'big')
    rng = Random(seed)
    timing: dict[str, Any] = {}
    t0 = asyncio.get_running_loop().time()
    out_share = await run_cubempc(node, circuit, input_values, input_dealers, instance_id, rng=rng, randomness_mode=randomness_mode, timing=timing, recv_timeout=180.0)
    timing['protocol_wall_ms'] = (asyncio.get_running_loop().time() - t0) * 1000.0
    out_layer = _find_output_layer(node, instance_id, circuit.output)
    await node.stop_server()
    return MPCNodeResult(rank=rank, pid=os.getpid(), output_share=out_share, output_layer=out_layer, metrics=_metrics_to_dict(node.metrics), timing=timing)

def run_cubempc_multiprocess(circuit: Circuit, input_values: dict[str, int], input_dealers: dict[str, int], *, n: int | None=None, t: int | None=None, base_port: int=23000, instance_id: str='mpc-test', startup_delay: float=3.0, timeout: float=600.0, randomness_mode: str='local', hosts: tuple[str, ...] | None=None) -> dict[str, Any]:
    import multiprocessing as mp
    import time
    from queue import Empty
    from cubempc.config import MPCConfig
    from cubempc.metrics import mpc_logical_cost
    from cubempc.network.coordinator import _metrics_from_dict, _metrics_to_dict
    if n is None:
        n = max(len(input_dealers), 5)
    if randomness_mode not in {'local', 'rg'}:
        raise ValueError(f'unsupported randomness_mode {randomness_mode!r}')
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    expected_plain = evaluate_plain(circuit, input_values)
    ctx = mp.get_context('spawn')
    out_queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []
    for rank in range(n):
        proc = ctx.Process(target=_mpc_worker_entry, args=(rank, n, t, base_port, circuit, input_values, input_dealers, instance_id, startup_delay, randomness_mode, hosts, out_queue), name=f'mpc-{rank}')
        proc.start()
        processes.append(proc)
    deadline = time.time() + timeout
    results: list[MPCNodeResult] = []
    while len(results) < n and time.time() < deadline:
        try:
            item = out_queue.get(timeout=0.5)
            if isinstance(item, MPCNodeResult):
                results.append(item)
            elif isinstance(item, dict) and 'error' in item:
                raise RuntimeError(f'MPC node {item.get('rank')} failed: {item['error']}')
        except Empty:
            continue
    for proc in processes:
        proc.join(timeout=15)
        if proc.exitcode != 0:
            raise RuntimeError(f'process exit code {proc.exitcode}')
    if len(results) < n:
        raise RuntimeError(f'expected {n} MPC results, got {len(results)}')
    results.sort(key=lambda r: r.rank)
    shares = {party_id(r.rank): r.output_share for r in results}
    reconstructed = reconstruct_output(shares, mpc.t)
    agg = __import__('cubempc.metrics', fromlist=['Metrics']).Metrics()
    for r in results:
        agg.merge(_metrics_from_dict(r.metrics))
    logical_metrics = mpc_logical_cost(circuit, n, mpc.t, randomness_mode=randomness_mode)
    protocol_equiv_latency_ms = _aggregate_protocol_latency_ms(results)
    protocol_latency_ms = max((float(r.timing.get('protocol_wall_ms', 0.0)) for r in results), default=protocol_equiv_latency_ms)
    return {'n': n, 't': mpc.t, 'expected_plain': expected_plain, 'reconstructed': reconstructed, 'shares': shares, 'per_rank': [r.__dict__ for r in results], 'aggregated_metrics': _metrics_to_dict(agg), 'logical_metrics': logical_metrics, 'protocol_latency_ms': protocol_latency_ms, 'protocol_equiv_latency_ms': protocol_equiv_latency_ms, 'randomness_mode': randomness_mode, 'match': reconstructed == expected_plain}

def _mpc_worker_entry(*args: object) -> None:
    import asyncio
    import os
    import traceback
    from queue import Queue
    rank, n, t, base_port, circuit, input_values, input_dealers, instance_id, startup_delay, randomness_mode, hosts, out_queue = args
    q: Queue = out_queue
    try:
        result = asyncio.run(_run_mpc_node_async(int(rank), int(n), t, int(base_port), circuit, input_values, input_dealers, str(instance_id), float(startup_delay), str(randomness_mode), hosts if hosts is None else tuple(hosts)))
        q.put(result)
    except Exception:
        q.put({'rank': rank, 'error': traceback.format_exc(), 'pid': os.getpid()})
run_multiplication = run_cubempc_multiprocess