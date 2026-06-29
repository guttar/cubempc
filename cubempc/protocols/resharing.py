from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from random import Random
from typing import Any
from cubempc.field import mod
from cubempc.messages import Message
from cubempc.network.node import InboxKey
from cubempc.poly import UniPoly
from cubempc.rs_decode import berlekamp_welch_decode
from cubempc.shamir import robust_reconstruct, shamir_share
PROTOCOL = 'rs'
Node = Any

def party_id(rank: int) -> int:
    return rank + 1

def rank_for_party(party: int) -> int:
    return party - 1

def _msg_filter(instance_id: str, dst_layer: int, msg_type: str, *, src_layer: int | None=None) -> Any:

    def _fn(m: Message) -> bool:
        if m.protocol != PROTOCOL or m.instance_id != instance_id:
            return False
        if m.msg_type != msg_type:
            return False
        if m.dst_layer != dst_layer:
            return False
        if src_layer is not None and m.src_layer != src_layer:
            return False
        return True
    return _fn

def _inbox_key(instance_id: str, dst_layer: int, msg_type: str, receiver_rank: int) -> InboxKey:
    return (dst_layer, PROTOCOL, instance_id, msg_type, receiver_rank)

def build_row_polynomial(s_i: int, random_shares: list[int], n: int) -> UniPoly:
    t = len(random_shares)
    points: list[tuple[int, int]] = [(0, mod(s_i))]
    for ell in range(1, t + 1):
        points.append((n + ell, mod(random_shares[ell - 1])))
    return UniPoly.interpolate(points)

async def run_resharing(node: Node, layer: int, input_share_key: Any, random_share_keys: list[Any], output_share_key: Any, instance_id: str, *, byzantine_sender_rank: int | None=None, recv_timeout: float=60.0, timing: dict[str, float] | None=None) -> int:
    n = node.n
    t = node.mpc_config.t
    if len(random_share_keys) != t:
        raise ValueError(f'need {t} random share keys, got {len(random_share_keys)}')
    layer_in = node.get_state(layer)
    s_i = mod(layer_in.values[input_share_key])
    random_shares = [mod(layer_in.values[key]) for key in random_share_keys]
    construct_cpu_start_ns = time.process_time_ns()
    construct_wall_start_ns = time.perf_counter_ns()
    row_i = build_row_polynomial(s_i, random_shares, n)
    my_party = party_id(node.rank)
    dst_layer = layer + 1
    if timing is not None:
        timing['rs_construct_cpu_ms'] = (time.process_time_ns() - construct_cpu_start_ns) / 1000000.0
        timing['rs_construct_wall_ms'] = (time.perf_counter_ns() - construct_wall_start_ns) / 1000000.0
    send_cpu_start_ns = time.process_time_ns()
    send_wall_start_ns = time.perf_counter_ns()
    for dst_rank in range(n):
        j_party = party_id(dst_rank)
        mu_ij = row_i.eval(j_party)
        if byzantine_sender_rank is not None and node.rank == byzantine_sender_rank:
            mu_ij = mod(mu_ij + 1)
        await node.send_layer_msg(layer, dst_layer, dst_rank, PROTOCOL, instance_id, 'rs_mu', {'src_party': my_party, 'dst_party': j_party, 'mu': mu_ij})
    if timing is not None:
        timing['rs_send_cpu_ms'] = (time.process_time_ns() - send_cpu_start_ns) / 1000000.0
        timing['rs_send_wall_ms'] = (time.perf_counter_ns() - send_wall_start_ns) / 1000000.0
    decode_cpu_start_ns = time.process_time_ns()
    decode_wall_start_ns = time.perf_counter_ns()
    msgs = await node.recv_until(_msg_filter(instance_id, dst_layer, 'rs_mu', src_layer=layer), n, recv_timeout, inbox_key=_inbox_key(instance_id, dst_layer, 'rs_mu', node.rank))
    mus: dict[int, int] = {}
    for m in msgs:
        payload = m.payload
        assert isinstance(payload, dict)
        if int(payload['dst_party']) != my_party:
            continue
        src_party = int(payload['src_party'])
        mus[src_party] = mod(int(payload['mu']))
    layer_out = node.get_state(dst_layer)
    ps = layer_out.protocol_states.setdefault(instance_id, {})
    if len(mus) < n:
        if timing is not None:
            timing['rs_decode_cpu_ms'] = (time.process_time_ns() - decode_cpu_start_ns) / 1000000.0
            timing['rs_decode_wall_ms'] = (time.perf_counter_ns() - decode_wall_start_ns) / 1000000.0
            output_cpu_start_ns = time.process_time_ns()
            output_wall_start_ns = time.perf_counter_ns()
        layer_out.values[output_share_key] = 0
        ps['success'] = False
        if timing is not None:
            timing['rs_output_cpu_ms'] = (time.process_time_ns() - output_cpu_start_ns) / 1000000.0
            timing['rs_output_wall_ms'] = (time.perf_counter_ns() - output_wall_start_ns) / 1000000.0
        return 0
    points = [(i, mus[i]) for i in sorted(mus.keys())]
    poly_x = berlekamp_welch_decode(points, t, t)
    if timing is not None:
        timing['rs_decode_cpu_ms'] = (time.process_time_ns() - decode_cpu_start_ns) / 1000000.0
        timing['rs_decode_wall_ms'] = (time.perf_counter_ns() - decode_wall_start_ns) / 1000000.0
        output_cpu_start_ns = time.process_time_ns()
        output_wall_start_ns = time.perf_counter_ns()
    if poly_x is None:
        new_share = 0
        ps['success'] = False
    else:
        new_share = poly_x.eval(0)
        ps['success'] = True
    layer_out.values[output_share_key] = new_share
    if timing is not None:
        timing['rs_output_cpu_ms'] = (time.process_time_ns() - output_cpu_start_ns) / 1000000.0
        timing['rs_output_wall_ms'] = (time.perf_counter_ns() - output_wall_start_ns) / 1000000.0
    return new_share

def setup_layer_shares_for_test(node: Node, layer: int, secret: int, instance_id: str, rng: Random) -> tuple[Any, list[Any], Any]:
    n = node.n
    t = node.mpc_config.t
    state = node.get_state(layer)
    input_share_key = (instance_id, 's_in')
    output_share_key = (instance_id, 's_out')
    random_share_keys: list[Any] = []
    shares_s = shamir_share(mod(secret), n, t, rng=rng)
    state.values[input_share_key] = shares_s[party_id(node.rank)]
    from cubempc.field import rand_field
    from cubempc.protocols.rg import map_rg_output_to_rs_keys
    rg_shares: list[int] = []
    for _ell in range(t):
        r_secret = rand_field(rng)
        shares_r = shamir_share(r_secret, n, t, rng=rng)
        rg_shares.append(shares_r[party_id(node.rank)])
    random_share_keys = map_rg_output_to_rs_keys(node, layer, instance_id, rg_shares, t)
    return (input_share_key, random_share_keys, output_share_key)

@dataclass
class RSNodeResult:
    rank: int
    pid: int
    old_share: int
    new_share: int
    success: bool
    metrics: dict[str, int]
    protocol_time_ms: float = 0.0

async def _run_rs_node_async(rank: int, n: int, t: int | None, base_port: int, layer: int, secret: int, instance_id: str, startup_delay: float, byzantine_sender_rank: int | None, hosts: tuple[str, ...] | None=None) -> RSNodeResult:
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
    seed = int.from_bytes(hashlib.sha256(f'{instance_id}:setup'.encode()).digest()[:8], 'big')
    rng = Random(seed)
    input_key, random_keys, output_key = setup_layer_shares_for_test(node, layer, secret, instance_id, rng)
    old_share = node.get_state(layer).values[input_key]
    t0 = asyncio.get_running_loop().time()
    new_share = await run_resharing(node, layer, input_key, random_keys, output_key, instance_id, byzantine_sender_rank=byzantine_sender_rank)
    protocol_time_ms = (asyncio.get_running_loop().time() - t0) * 1000.0
    success = bool(node.get_state(layer + 1).protocol_states.get(instance_id, {}).get('success', False))
    await node.stop_server()
    return RSNodeResult(rank=rank, pid=os.getpid(), old_share=old_share, new_share=new_share, success=success, metrics=_metrics_to_dict(node.metrics), protocol_time_ms=protocol_time_ms)

def run_resharing_multiprocess(n: int, secret: int, *, t: int | None=None, base_port: int=21000, layer: int=0, instance_id: str='rs-test', startup_delay: float=2.0, timeout: float=120.0, byzantine_sender_rank: int | None=None, hosts: tuple[str, ...] | None=None) -> dict[str, Any]:
    import multiprocessing as mp
    import time
    from queue import Empty
    from cubempc.config import MPCConfig
    from cubempc.network.coordinator import _metrics_from_dict, _metrics_to_dict
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    ctx = mp.get_context('spawn')
    out_queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []
    for rank in range(n):
        proc = ctx.Process(target=_rs_worker_entry, args=(rank, n, t, base_port, layer, secret, instance_id, startup_delay, byzantine_sender_rank, hosts, out_queue), name=f'rs-{rank}')
        proc.start()
        processes.append(proc)
    deadline = time.time() + timeout
    results: list[RSNodeResult] = []
    while len(results) < n and time.time() < deadline:
        try:
            item = out_queue.get(timeout=0.5)
            if isinstance(item, RSNodeResult):
                results.append(item)
            elif isinstance(item, dict) and 'error' in item:
                raise RuntimeError(f'RS node {item.get('rank')} failed: {item['error']}')
        except Empty:
            continue
    from cubempc.network.coordinator import join_worker_processes
    join_worker_processes(processes)
    if len(results) < n:
        raise RuntimeError(f'expected {n} RS results, got {len(results)}')
    results.sort(key=lambda r: r.rank)
    old_shares = {party_id(r.rank): r.old_share for r in results}
    new_shares = {party_id(r.rank): r.new_share for r in results}
    reconstructed = robust_reconstruct(new_shares, mpc.t, mpc.t)
    agg = __import__('cubempc.metrics', fromlist=['Metrics']).Metrics()
    for r in results:
        agg.merge(_metrics_from_dict(r.metrics))
    total_time_ms = max((r.protocol_time_ms for r in results)) if results else 0.0
    return {'n': n, 't': mpc.t, 'secret': mod(secret), 'reconstructed': reconstructed, 'old_shares': old_shares, 'new_shares': new_shares, 'all_success': all((r.success for r in results)), 'per_rank': [r.__dict__ for r in results], 'aggregated_metrics': _metrics_to_dict(agg), 'total_time_ms': total_time_ms}

def _rs_worker_entry(*args: object) -> None:
    import asyncio
    import os
    import traceback
    from queue import Queue
    rank, n, t, base_port, layer, secret, instance_id, startup_delay, byzantine_sender_rank, hosts, out_queue = args
    q: Queue = out_queue
    try:
        result = asyncio.run(_run_rs_node_async(int(rank), int(n), t, int(base_port), int(layer), int(secret), str(instance_id), float(startup_delay), byzantine_sender_rank if byzantine_sender_rank is None else int(byzantine_sender_rank), hosts if hosts is None else tuple(hosts)))
        q.put(result)
    except Exception:
        q.put({'rank': rank, 'error': traceback.format_exc(), 'pid': os.getpid()})