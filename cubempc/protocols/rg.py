from __future__ import annotations
import asyncio
import hashlib
from dataclasses import dataclass
from random import Random
from typing import Any
from cubempc.algebra_cache import vandermonde_matrix as cached_vandermonde_matrix
from cubempc.field import add, mod, mul, rand_field
from cubempc.profiling import bump as profile_bump
from cubempc.protocols.vss import party_id, run_discard_vss_warmup, run_vss, run_vss_batch
from cubempc.shamir import robust_reconstruct
Node = Any
_rg_single_call_counter: int = 0

def reset_rg_single_call_counter() -> None:
    global _rg_single_call_counter
    _rg_single_call_counter = 0

def get_rg_single_call_counter() -> int:
    return _rg_single_call_counter

@dataclass
class RandomnessRecord:
    layer_id: int
    gate_index: int
    values: list[int]
    metadata: dict[str, Any] | None = None

def _vandermonde_matrix(n: int, q: int) -> list[list[int]]:
    return cached_vandermonde_matrix(n, q)

def combine_vandermonde_shares(shares_of_m: dict[int, int], n: int, q: int) -> list[int]:
    matrix = _vandermonde_matrix(n, q)
    output: list[int] = []
    for ell in range(q):
        acc = 0
        for i in range(1, n + 1):
            acc = add(acc, mul(matrix[i - 1][ell], shares_of_m[i]))
        output.append(mod(acc))
    return output

def map_rg_output_to_rs_keys(node: Node, layer: int, instance_id: str, rg_shares: list[int], t: int) -> list[tuple[str, str, int]]:
    if len(rg_shares) < t:
        raise ValueError(f'need at least {t} RG shares for RS, got {len(rg_shares)}')
    state = node.get_state(layer)
    keys: list[tuple[str, str, int]] = []
    for ell in range(1, t + 1):
        key = (instance_id, 'rs_rand', ell)
        state.values[key] = mod(rg_shares[ell - 1])
        keys.append(key)
    return keys

def map_rg_output_to_mul_keys(node: Node, layer: int, instance_id: str, rg_shares: list[int], t: int) -> list[tuple[str, str, int]]:
    need = 3 * t + 1
    if len(rg_shares) < need:
        raise ValueError(f'need at least {need} RG shares for MUL, got {len(rg_shares)}')
    state = node.get_state(layer)
    keys: list[tuple[str, str, int]] = []
    for ell in range(need):
        key = (instance_id, 'mul_rand', ell)
        state.values[key] = mod(rg_shares[ell])
        keys.append(key)
    return keys

async def run_rg(node: Node, start_layer: int, q: int | None, instance_id: str, rng: Random | None=None, *, warmup_round: bool=False, timing: dict[str, Any] | None=None) -> list[int]:
    n = node.n
    t = node.mpc_config.t
    if q is None:
        q = n - t
    max_q = n - t
    if q < 1 or q > max_q:
        raise ValueError(f'need 1 <= q <= n-t ({max_q}), got q={q}, n={n}, t={t}')
    if warmup_round:
        await run_discard_vss_warmup(node, start_layer, instance_id, rng)
    global _rg_single_call_counter
    _rg_single_call_counter += 1
    profile_bump(rg_call_count=1, rg_vss_instance_count=n, rg_vss_dealer_count=n)
    loop = asyncio.get_running_loop()
    wall_t0 = loop.time()
    shares_of_m: dict[int, int] = {}
    vss_round_ms: list[float] = [0.0] * n

    async def _run_dealer_vss(dealer_rank: int) -> tuple[int, int, float]:
        vss_id = f'{instance_id}:vss:{dealer_rank}'
        dealer_seed = int.from_bytes(hashlib.sha256(f'{vss_id}:rank:{node.rank}'.encode()).digest()[:8], 'big')
        dealer_rng = Random(dealer_seed)
        if node.rank == dealer_rank:
            m_i = rand_field(dealer_rng)
        else:
            m_i = 0
        t0 = loop.time()
        share_i = await run_vss(node=node, start_layer=start_layer, dealer_rank=dealer_rank, secret=m_i, instance_id=vss_id, rng=dealer_rng)
        return (dealer_rank, share_i, (loop.time() - t0) * 1000.0)
    dealer_results = await asyncio.gather(*(_run_dealer_vss(dealer_rank) for dealer_rank in range(n)))
    for dealer_rank, share_i, elapsed_ms in sorted(dealer_results, key=lambda item: item[0]):
        vss_round_ms[dealer_rank] = elapsed_ms
        shares_of_m[dealer_rank + 1] = share_i
    t0 = loop.time()
    output = combine_vandermonde_shares(shares_of_m, n, q)
    combine_ms = (loop.time() - t0) * 1000.0
    wall_ms = (loop.time() - wall_t0) * 1000.0
    state = node.get_state(start_layer + 3)
    state.values[instance_id, 'rg_shares'] = output
    if timing is not None:
        timing['vss_round_ms'] = vss_round_ms
        timing['combine_ms'] = combine_ms
        timing['wall_ms'] = wall_ms
        timing['parallel_equiv_ms'] = wall_ms
        timing['sequential_impl_ms'] = sum(vss_round_ms) + combine_ms
    return output

async def run_rg_batch(node: Node, start_layer: int, batch_size: int, instance_id: str, *, layer_id: int=0, purpose: str='mul_mask', q: int | None=None, rng: Random | None=None, warmup_round: bool=False, timing: dict[str, Any] | None=None) -> list[RandomnessRecord]:
    _ = purpose
    n = node.n
    t = node.mpc_config.t
    if batch_size < 1:
        raise ValueError(f'batch_size must be >= 1, got {batch_size}')
    if q is None:
        q = n - t
    max_q = n - t
    if q < 1 or q > max_q:
        raise ValueError(f'need 1 <= q <= n-t ({max_q}), got q={q}, n={n}, t={t}')
    if warmup_round:
        await run_discard_vss_warmup(node, start_layer, instance_id, rng)
    profile_bump(rg_batch_count=1, rg_vss_instance_count=n, rg_vss_dealer_count=n, rg_vss_secret_count=batch_size, rg_output_count=batch_size)
    loop = asyncio.get_running_loop()
    wall_t0 = loop.time()
    shares_of_m: dict[int, list[int]] = {}
    vss_round_ms: list[float] = [0.0] * n

    async def _run_dealer_vss_batch(dealer_rank: int) -> tuple[int, list[int], float]:
        vss_id = f'{instance_id}:vss:{dealer_rank}'
        dealer_seed = int.from_bytes(hashlib.sha256(f'{vss_id}:rank:{node.rank}'.encode()).digest()[:8], 'big')
        dealer_rng = Random(dealer_seed)
        secrets = [rand_field(dealer_rng) if node.rank == dealer_rank else 0 for _ in range(batch_size)]
        t0 = loop.time()
        share_list = await run_vss_batch(node=node, start_layer=start_layer, dealer_rank=dealer_rank, secrets=secrets, instance_id=vss_id, rng=dealer_rng)
        return (dealer_rank, share_list, (loop.time() - t0) * 1000.0)
    dealer_results = await asyncio.gather(*(_run_dealer_vss_batch(dealer_rank) for dealer_rank in range(n)))
    for dealer_rank, share_list, elapsed_ms in sorted(dealer_results, key=lambda item: item[0]):
        vss_round_ms[dealer_rank] = elapsed_ms
        shares_of_m[dealer_rank + 1] = share_list
    t0 = loop.time()
    records: list[RandomnessRecord] = []
    for gate_index in range(batch_size):
        gate_shares = {party: shares_of_m[party][gate_index] for party in range(1, n + 1)}
        output = combine_vandermonde_shares(gate_shares, n, q)
        records.append(RandomnessRecord(layer_id=layer_id, gate_index=gate_index, values=output, metadata={'purpose': purpose, 'q': q}))
    combine_ms = (loop.time() - t0) * 1000.0
    wall_ms = (loop.time() - wall_t0) * 1000.0
    state = node.get_state(start_layer + 3)
    state.values[instance_id, 'rg_batch_shares'] = records
    if timing is not None:
        timing['vss_round_ms'] = vss_round_ms
        timing['combine_ms'] = combine_ms
        timing['wall_ms'] = wall_ms
        timing['batch_size'] = batch_size
    return records

@dataclass
class RGNodeResult:
    rank: int
    pid: int
    shares: list[int]
    metrics: dict[str, int]
    protocol_time_ms: float = 0.0

@dataclass
class RGNodeDoubleResult:
    rank: int
    pid: int
    shares_a: list[int]
    shares_b: list[int]
    metrics: dict[str, int]
    protocol_time_ms: float = 0.0

async def _run_rg_node_async(rank: int, n: int, t: int | None, base_port: int, start_layer: int, q: int | None, instance_id: str, startup_delay: float, warmup_round: bool, hosts: tuple[str, ...] | None=None) -> RGNodeResult:
    import os
    from cubempc.config import MPCConfig
    from cubempc.network.node import NodeProcess
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    node = NodeProcess(rank, n, base_port, mpc)
    await node.start_server()
    await asyncio.sleep(startup_delay)
    await node.connect_peers()
    import hashlib
    seed = int.from_bytes(hashlib.sha256(f'{instance_id}:{rank}'.encode()).digest()[:8], 'big')
    rng = Random(seed)
    t0 = asyncio.get_running_loop().time()
    shares = await run_rg(node, start_layer, q, instance_id, rng=rng, warmup_round=warmup_round)
    protocol_time_ms = (asyncio.get_running_loop().time() - t0) * 1000.0
    await node.stop_server()
    from cubempc.network.coordinator import _metrics_to_dict
    return RGNodeResult(rank=rank, pid=os.getpid(), shares=shares, metrics=_metrics_to_dict(node.metrics), protocol_time_ms=protocol_time_ms)

async def _run_rg_node_double_async(rank: int, n: int, t: int | None, base_port: int, start_layer: int, q: int | None, instance_id: str, startup_delay: float, hosts: tuple[str, ...] | None=None) -> RGNodeDoubleResult:
    import os
    from cubempc.config import MPCConfig
    from cubempc.network.node import NodeProcess
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    node = NodeProcess(rank, n, base_port, mpc)
    await node.start_server()
    await asyncio.sleep(startup_delay)
    await node.connect_peers()
    import hashlib
    t0 = asyncio.get_running_loop().time()
    shares_a = await run_rg(node, start_layer, q, f'{instance_id}:a', rng=Random(int.from_bytes(hashlib.sha256(f'{instance_id}:a:{rank}'.encode()).digest()[:8], 'big')))
    shares_b = await run_rg(node, start_layer, q, f'{instance_id}:b', rng=Random(int.from_bytes(hashlib.sha256(f'{instance_id}:b:{rank}'.encode()).digest()[:8], 'big')))
    protocol_time_ms = (asyncio.get_running_loop().time() - t0) * 1000.0
    await node.stop_server()
    from cubempc.network.coordinator import _metrics_to_dict
    return RGNodeDoubleResult(rank=rank, pid=os.getpid(), shares_a=shares_a, shares_b=shares_b, metrics=_metrics_to_dict(node.metrics), protocol_time_ms=protocol_time_ms)

def _reconstruct_rg_shares(per_rank_shares: dict[int, list[int]], q: int, t: int) -> list[int | None]:
    reconstructed: list[int | None] = []
    for ell in range(q):
        shares_ell = {party: per_rank_shares[party][ell] for party in per_rank_shares}
        reconstructed.append(robust_reconstruct(shares_ell, t, t))
    return reconstructed

def run_rg_multiprocess(n: int, *, t: int | None=None, q: int | None=None, base_port: int=20000, start_layer: int=0, instance_id: str='rg-test', startup_delay: float=2.0, warmup_round: bool=False, timeout: float=300.0, hosts: tuple[str, ...] | None=None) -> dict[str, Any]:
    import multiprocessing as mp
    import time
    from queue import Empty
    from cubempc.config import MPCConfig
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    if q is None:
        q = n - mpc.t
    ctx = mp.get_context('spawn')
    out_queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []
    for rank in range(n):
        proc = ctx.Process(target=_rg_worker_entry, args=(rank, n, t, base_port, start_layer, q, instance_id, startup_delay, warmup_round, hosts, out_queue), name=f'rg-{rank}')
        proc.start()
        processes.append(proc)
    deadline = time.time() + timeout
    results: list[RGNodeResult] = []
    while len(results) < n and time.time() < deadline:
        try:
            item = out_queue.get(timeout=0.5)
            if isinstance(item, RGNodeResult):
                results.append(item)
            elif isinstance(item, dict) and 'error' in item:
                raise RuntimeError(f'RG node {item.get('rank')} failed: {item['error']}')
        except Empty:
            continue
    from cubempc.network.coordinator import join_worker_processes
    join_worker_processes(processes)
    if len(results) < n:
        raise RuntimeError(f'expected {n} RG results, got {len(results)}')
    results.sort(key=lambda r: r.rank)
    per_rank_shares = {party_id(r.rank): r.shares for r in results}
    reconstructed = _reconstruct_rg_shares(per_rank_shares, q, mpc.t)
    from cubempc.metrics import Metrics
    from cubempc.network.coordinator import _metrics_from_dict, _metrics_to_dict
    agg = Metrics()
    for r in results:
        agg.merge(_metrics_from_dict(r.metrics))
    total_time_ms = max((r.protocol_time_ms for r in results)) if results else 0.0
    return {'n': n, 't': mpc.t, 'q': q, 'reconstructed': reconstructed, 'per_rank_shares': per_rank_shares, 'per_rank': [r.__dict__ for r in results], 'aggregated_metrics': _metrics_to_dict(agg), 'total_time_ms': total_time_ms}

def run_rg_twice_multiprocess(n: int, *, t: int | None=None, q: int | None=None, base_port: int=20000, start_layer: int=0, instance_id: str='rg-twice', startup_delay: float=2.0, timeout: float=600.0, hosts: tuple[str, ...] | None=None) -> dict[str, Any]:
    import multiprocessing as mp
    import time
    from queue import Empty
    from cubempc.config import MPCConfig
    from cubempc.metrics import Metrics
    from cubempc.network.coordinator import _metrics_from_dict, _metrics_to_dict
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    if q is None:
        q = n - mpc.t
    ctx = mp.get_context('spawn')
    out_queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []
    for rank in range(n):
        proc = ctx.Process(target=_rg_double_worker_entry, args=(rank, n, t, base_port, start_layer, q, instance_id, startup_delay, hosts, out_queue), name=f'rg2-{rank}')
        proc.start()
        processes.append(proc)
    deadline = time.time() + timeout
    results: list[RGNodeDoubleResult] = []
    while len(results) < n and time.time() < deadline:
        try:
            item = out_queue.get(timeout=0.5)
            if isinstance(item, RGNodeDoubleResult):
                results.append(item)
            elif isinstance(item, dict) and 'error' in item:
                raise RuntimeError(f'RG node {item.get('rank')} failed: {item['error']}')
        except Empty:
            continue
    from cubempc.network.coordinator import join_worker_processes
    join_worker_processes(processes)
    if len(results) < n:
        raise RuntimeError(f'expected {n} RG results, got {len(results)}')
    results.sort(key=lambda r: r.rank)
    shares_a = {party_id(r.rank): r.shares_a for r in results}
    shares_b = {party_id(r.rank): r.shares_b for r in results}
    reconstructed_a = _reconstruct_rg_shares(shares_a, q, mpc.t)
    reconstructed_b = _reconstruct_rg_shares(shares_b, q, mpc.t)
    agg = Metrics()
    for r in results:
        agg.merge(_metrics_from_dict(r.metrics))
    total_time_ms = max((r.protocol_time_ms for r in results)) if results else 0.0
    return {'n': n, 't': mpc.t, 'q': q, 'reconstructed_a': reconstructed_a, 'reconstructed_b': reconstructed_b, 'per_rank': [r.__dict__ for r in results], 'aggregated_metrics': _metrics_to_dict(agg), 'total_time_ms': total_time_ms}

def _rg_worker_entry(*args: object) -> None:
    import asyncio
    import os
    import traceback
    from queue import Queue
    rank, n, t, base_port, start_layer, q, instance_id, startup_delay, warmup_round, hosts, out_queue = args
    q_out: Queue = out_queue
    try:
        result = asyncio.run(_run_rg_node_async(int(rank), int(n), t, int(base_port), int(start_layer), q, str(instance_id), float(startup_delay), bool(warmup_round), hosts if hosts is None else tuple(hosts)))
        q_out.put(result)
    except Exception:
        q_out.put({'rank': rank, 'error': traceback.format_exc(), 'pid': os.getpid()})

def _rg_double_worker_entry(*args: object) -> None:
    import asyncio
    import os
    import traceback
    from queue import Queue
    rank, n, t, base_port, start_layer, q, instance_id, startup_delay, hosts, out_queue = args
    q_out: Queue = out_queue
    try:
        result = asyncio.run(_run_rg_node_double_async(int(rank), int(n), t, int(base_port), int(start_layer), q, str(instance_id), float(startup_delay), hosts if hosts is None else tuple(hosts)))
        q_out.put(result)
    except Exception:
        q_out.put({'rank': rank, 'error': traceback.format_exc(), 'pid': os.getpid()})

def run_randomness_generation(nodes: list[Any], config: Any, *, instance_id: str) -> Any:
    _ = (nodes, config, instance_id)
    raise NotImplementedError('use run_rg / run_rg_multiprocess')