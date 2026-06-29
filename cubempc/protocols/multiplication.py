from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from random import Random
from typing import Any
from cubempc.field import add, mod, mul, pow_mod
from cubempc.messages import Message
from cubempc.network.node import InboxKey
from cubempc.protocols.resharing import run_resharing
from cubempc.rs_decode import berlekamp_welch_decode
from cubempc.shamir import robust_reconstruct, shamir_share
PROTOCOL = 'mul'
MulPhase = str
Node = Any

def party_id(rank: int) -> int:
    return rank + 1

def rank_for_party(party: int) -> int:
    return party - 1

def _msg_filter(instance_id: str, layer: int, msg_type: str, *, src_layer: int | None=None) -> Any:

    def _fn(m: Message) -> bool:
        if m.protocol != PROTOCOL or m.instance_id != instance_id:
            return False
        if m.msg_type != msg_type:
            return False
        if m.dst_layer != layer:
            return False
        if src_layer is not None and m.src_layer != src_layer:
            return False
        return True
    return _fn

def _inbox_key(instance_id: str, layer: int, msg_type: str, receiver_rank: int) -> InboxKey:
    return (layer, PROTOCOL, instance_id, msg_type, receiver_rank)

def resharing_prep_keys(instance_id: str, t: int) -> list[tuple[str, str, int]]:
    return [(instance_id, 'rs_prep', ell) for ell in range(t)]

def mask_st_share_at_j(r_local: list[int], j_party: int, t: int) -> int:
    acc = mod(r_local[0])
    for ell in range(1, t + 1):
        acc = add(acc, mul(mod(r_local[ell]), pow_mod(j_party, ell)))
    return acc

def mask_s2t_share_at_j(r_local: list[int], j_party: int, t: int) -> int:
    acc = mod(r_local[0])
    for ell in range(1, 2 * t + 1):
        acc = add(acc, mul(mod(r_local[t + ell]), pow_mod(j_party, ell)))
    return acc

def reconstruct_mask_value(shares_by_party: dict[int, int], degree: int, max_errors: int) -> int | None:
    if not shares_by_party:
        return None
    return robust_reconstruct(shares_by_party, degree, max_errors)

async def run_mul(node: Node, layer: int, x_key: Any, y_key: Any, random_keys: list[Any], output_key: Any, instance_id: str, *, recv_timeout: float=60.0, timing: dict[str, float] | None=None, phase: MulPhase='full') -> int:
    loop = asyncio.get_running_loop()
    n = node.n
    t = node.mpc_config.t
    need_random = 3 * t + 1
    if len(random_keys) != need_random:
        raise ValueError(f'need {need_random} random keys (r_0..r_{{3t}}), got {len(random_keys)}')
    if len(set(random_keys)) != len(random_keys):
        raise ValueError('random keys for MUL must be distinct')
    if phase not in {'full', 'mask_prep', 'online'}:
        raise ValueError(f'unknown mul phase: {phase}')
    state = node.get_state(layer)
    my_party = party_id(node.rank)
    if phase in {'full', 'mask_prep'}:
        x_i = mod(state.values[x_key])
        y_i = mod(state.values[y_key])
        r_local = [mod(state.values[k]) for k in random_keys]
        compute_cpu_start_ns = time.process_time_ns()
        compute_wall_start_ns = time.perf_counter_ns()
        mask_payloads: list[tuple[int, dict[str, int]]] = []
        for dst_rank in range(n):
            j_party = party_id(dst_rank)
            s_t = mask_st_share_at_j(r_local, j_party, t)
            s_2t = mask_s2t_share_at_j(r_local, j_party, t)
            mask_payloads.append((dst_rank, {'src_party': my_party, 'dst_party': j_party, 'S_t_share': s_t, 'S_2t_share': s_2t}))
        if timing is not None:
            timing['mul_mask_compute_cpu_ms'] = (time.process_time_ns() - compute_cpu_start_ns) / 1000000.0
            timing['mul_mask_compute_wall_ms'] = (time.perf_counter_ns() - compute_wall_start_ns) / 1000000.0
        send_cpu_start_ns = time.process_time_ns()
        send_wall_start_ns = time.perf_counter_ns()
        for dst_rank, payload in mask_payloads:
            await node.send_layer_msg(layer, layer, dst_rank, PROTOCOL, instance_id, 'mul_mask_eval', payload)
        if timing is not None:
            timing['mul_mask_send_cpu_ms'] = (time.process_time_ns() - send_cpu_start_ns) / 1000000.0
            timing['mul_mask_send_wall_ms'] = (time.perf_counter_ns() - send_wall_start_ns) / 1000000.0
            timing['com0_mask_ms'] = timing['mul_mask_compute_wall_ms'] + timing['mul_mask_send_wall_ms']
        if phase == 'mask_prep':
            return 0
    online_t0 = loop.time()
    x_i = mod(state.values[x_key])
    y_i = mod(state.values[y_key])
    recon_cpu_start_ns = time.process_time_ns()
    recon_wall_start_ns = time.perf_counter_ns()
    mask_msgs = await node.recv_until(_msg_filter(instance_id, layer, 'mul_mask_eval', src_layer=layer), n, recv_timeout, inbox_key=_inbox_key(instance_id, layer, 'mul_mask_eval', node.rank))
    s_t_shares: dict[int, int] = {}
    s_2t_shares: dict[int, int] = {}
    for m in mask_msgs:
        p = m.payload
        assert isinstance(p, dict)
        if int(p['dst_party']) != my_party:
            continue
        src = int(p['src_party'])
        s_t_shares[src] = mod(int(p['S_t_share']))
        s_2t_shares[src] = mod(int(p['S_2t_share']))
    s_t_j = reconstruct_mask_value(s_t_shares, t, t)
    s_2t_j = reconstruct_mask_value(s_2t_shares, t, t)
    if timing is not None:
        timing['mul_mask_reconstruct_cpu_ms'] = (time.process_time_ns() - recon_cpu_start_ns) / 1000000.0
        timing['mul_mask_reconstruct_wall_ms'] = (time.perf_counter_ns() - recon_wall_start_ns) / 1000000.0
    if s_t_j is None or s_2t_j is None:
        node.get_state(layer + 1).values[output_key] = 0
        if timing is not None:
            timing['online_ms'] = (loop.time() - online_t0) * 1000.0
        return 0
    product_cpu_start_ns = time.process_time_ns()
    product_wall_start_ns = time.perf_counter_ns()
    d_j = sub_online_term(x_i, y_i, s_2t_j)
    if timing is not None:
        timing['mul_product_compute_cpu_ms'] = (time.process_time_ns() - product_cpu_start_ns) / 1000000.0
        timing['mul_product_compute_wall_ms'] = (time.perf_counter_ns() - product_wall_start_ns) / 1000000.0
    broadcast_cpu_start_ns = time.process_time_ns()
    broadcast_wall_start_ns = time.perf_counter_ns()
    await node.send_layer_msg(layer, layer, 0, PROTOCOL, instance_id, 'mul_d', {'party': my_party, 'd': d_j}, channel='broadcast')
    if timing is not None:
        timing['mul_product_broadcast_cpu_ms'] = (time.process_time_ns() - broadcast_cpu_start_ns) / 1000000.0
        timing['mul_product_broadcast_wall_ms'] = (time.perf_counter_ns() - broadcast_wall_start_ns) / 1000000.0
    public_cpu_start_ns = time.process_time_ns()
    public_wall_start_ns = time.perf_counter_ns()
    d_msgs = await node.recv_until(_msg_filter(instance_id, layer, 'mul_d', src_layer=layer), n, recv_timeout, inbox_key=_inbox_key(instance_id, layer, 'mul_d', node.rank))
    d_by_party: dict[int, int] = {}
    for m in d_msgs:
        p = m.payload
        assert isinstance(p, dict)
        d_by_party[int(p['party'])] = mod(int(p['d']))
    delta = decode_delta(d_by_party, t)
    if timing is not None:
        timing['mul_public_decode_cpu_ms'] = (time.process_time_ns() - public_cpu_start_ns) / 1000000.0
        timing['mul_public_decode_wall_ms'] = (time.perf_counter_ns() - public_wall_start_ns) / 1000000.0
    if delta is None:
        node.get_state(layer + 1).values[output_key] = 0
        if timing is not None:
            timing['online_ms'] = (loop.time() - online_t0) * 1000.0
        return 0
    r0_mask_key = (instance_id, 'r0_mask_share')
    r0_out_key = (instance_id, 'r0_reshared')
    state.values[r0_mask_key] = mod(s_t_j)
    rs_prep = resharing_prep_keys(instance_id, t)
    rs_timing: dict[str, float] = {}
    await run_resharing(node=node, layer=layer, input_share_key=r0_mask_key, random_share_keys=rs_prep, output_share_key=r0_out_key, instance_id=f'{instance_id}:reshare-r0', recv_timeout=recv_timeout, timing=rs_timing)
    if timing is not None:
        timing['mul_rs_construct_cpu_ms'] = rs_timing.get('rs_construct_cpu_ms', 0.0)
        timing['mul_rs_send_cpu_ms'] = rs_timing.get('rs_send_cpu_ms', 0.0)
        timing['mul_rs_decode_cpu_ms'] = rs_timing.get('rs_decode_cpu_ms', 0.0)
        timing['mul_rs_construct_wall_ms'] = rs_timing.get('rs_construct_wall_ms', 0.0)
        timing['mul_rs_send_wall_ms'] = rs_timing.get('rs_send_wall_ms', 0.0)
        timing['mul_rs_decode_wall_ms'] = rs_timing.get('rs_decode_wall_ms', 0.0)
    add_cpu_start_ns = time.process_time_ns()
    add_wall_start_ns = time.perf_counter_ns()
    r0_fresh = mod(node.get_state(layer + 1).values[r0_out_key])
    product_share = add(r0_fresh, delta)
    node.get_state(layer + 1).values[output_key] = product_share
    if timing is not None:
        timing['mul_output_add_cpu_ms'] = (time.process_time_ns() - add_cpu_start_ns) / 1000000.0
        timing['mul_output_add_wall_ms'] = (time.perf_counter_ns() - add_wall_start_ns) / 1000000.0
        timing['online_ms'] = (loop.time() - online_t0) * 1000.0
    return product_share

def sub_online_term(x_j: int, y_j: int, s_2t_j: int) -> int:
    return mod(mul(x_j, y_j) - s_2t_j)

def decode_delta(d_by_party: dict[int, int], t: int) -> int | None:
    if len(d_by_party) < 4 * t + 1:
        return None
    points = [(party, val) for party, val in sorted(d_by_party.items())]
    d_poly = berlekamp_welch_decode(points, 2 * t, t)
    if d_poly is None:
        return None
    return d_poly.eval(0)

def setup_mul_layer_for_test(node: Node, layer: int, x: int, y: int, instance_id: str, rng: Random) -> tuple[Any, Any, list[Any], list[Any], Any]:
    n = node.n
    t = node.mpc_config.t
    st = node.get_state(layer)
    x_key = (instance_id, 'x')
    y_key = (instance_id, 'y')
    output_key = (instance_id, 'xy_out')
    random_keys: list[Any] = []
    st.values[x_key] = shamir_share(mod(x), n, t, rng=rng)[party_id(node.rank)]
    st.values[y_key] = shamir_share(mod(y), n, t, rng=rng)[party_id(node.rank)]
    from cubempc.field import rand_field
    from cubempc.protocols.rg import map_rg_output_to_mul_keys
    rg_shares: list[int] = []
    for _ell in range(3 * t + 1):
        secret = rand_field(rng)
        rg_shares.append(shamir_share(secret, n, t, rng=rng)[party_id(node.rank)])
    random_keys = map_rg_output_to_mul_keys(node, layer, instance_id, rg_shares, t)
    rs_keys = resharing_prep_keys(instance_id, t)
    for key in rs_keys:
        secret = rand_field(rng)
        st.values[key] = shamir_share(secret, n, t, rng=rng)[party_id(node.rank)]
    return (x_key, y_key, random_keys, rs_keys, output_key)

@dataclass
class MulNodeResult:
    rank: int
    pid: int
    product_share: int
    metrics: dict[str, int]
    protocol_time_ms: float = 0.0

async def _run_mul_node_async(rank: int, n: int, t: int | None, base_port: int, layer: int, x: int, y: int, instance_id: str, startup_delay: float, hosts: tuple[str, ...] | None=None) -> MulNodeResult:
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
    seed = int.from_bytes(hashlib.sha256(f'{instance_id}:mul-setup'.encode()).digest()[:8], 'big')
    rng = Random(seed)
    x_key, y_key, random_keys, _rs_keys, output_key = setup_mul_layer_for_test(node, layer, x, y, instance_id, rng)
    t0 = asyncio.get_running_loop().time()
    product = await run_mul(node, layer, x_key, y_key, random_keys, output_key, instance_id)
    protocol_time_ms = (asyncio.get_running_loop().time() - t0) * 1000.0
    await node.stop_server()
    return MulNodeResult(rank=rank, pid=os.getpid(), product_share=product, metrics=_metrics_to_dict(node.metrics), protocol_time_ms=protocol_time_ms)

def run_mul_multiprocess(n: int, x: int, y: int, *, t: int | None=None, base_port: int=22000, layer: int=0, instance_id: str='mul-test', startup_delay: float=2.5, timeout: float=120.0, hosts: tuple[str, ...] | None=None) -> dict[str, Any]:
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
        proc = ctx.Process(target=_mul_worker_entry, args=(rank, n, t, base_port, layer, x, y, instance_id, startup_delay, hosts, out_queue), name=f'mul-{rank}')
        proc.start()
        processes.append(proc)
    deadline = time.time() + timeout
    results: list[MulNodeResult] = []
    while len(results) < n and time.time() < deadline:
        try:
            item = out_queue.get(timeout=0.5)
            if isinstance(item, MulNodeResult):
                results.append(item)
            elif isinstance(item, dict) and 'error' in item:
                raise RuntimeError(f'MUL node {item.get('rank')} failed: {item['error']}')
        except Empty:
            continue
    from cubempc.network.coordinator import join_worker_processes
    join_worker_processes(processes)
    if len(results) < n:
        raise RuntimeError(f'expected {n} MUL results, got {len(results)}')
    results.sort(key=lambda r: r.rank)
    shares = {party_id(r.rank): r.product_share for r in results}
    expected = mod(x * y)
    reconstructed = robust_reconstruct(shares, mpc.t, mpc.t)
    agg = __import__('cubempc.metrics', fromlist=['Metrics']).Metrics()
    for r in results:
        agg.merge(_metrics_from_dict(r.metrics))
    total_time_ms = max((r.protocol_time_ms for r in results)) if results else 0.0
    return {'n': n, 't': mpc.t, 'x': mod(x), 'y': mod(y), 'expected': expected, 'reconstructed': reconstructed, 'shares': shares, 'per_rank': [r.__dict__ for r in results], 'aggregated_metrics': _metrics_to_dict(agg), 'total_time_ms': total_time_ms}

def _mul_worker_entry(*args: object) -> None:
    import asyncio
    import os
    import traceback
    from queue import Queue
    rank, n, t, base_port, layer, x, y, instance_id, startup_delay, hosts, out_queue = args
    q: Queue = out_queue
    try:
        result = asyncio.run(_run_mul_node_async(int(rank), int(n), t, int(base_port), int(layer), int(x), int(y), str(instance_id), float(startup_delay), hosts if hosts is None else tuple(hosts)))
        q.put(result)
    except Exception:
        q.put({'rank': rank, 'error': traceback.format_exc(), 'pid': os.getpid()})
run_multiplication = run_mul_multiprocess