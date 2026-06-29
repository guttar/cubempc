from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from random import Random
from typing import Any
from cubempc.algebra_cache import lagrange_coefficients_for_points
from cubempc.batch_payload import parse_scalar_table, scalar_table_payload
from cubempc.field import add, mod, mul, rand_field
from cubempc.messages import Message
from cubempc.network.node import InboxKey
from cubempc.poly import BiPoly, TriPoly, UniPoly
from cubempc.profiling import bump as profile_bump
from cubempc.rs_decode import berlekamp_welch_decode, decode_poly_valued, decode_poly_valued_batch
from cubempc.shamir import robust_reconstruct
from cubempc.vss_profiling import get_profiler
PROTOCOL = 'vss'
DEFAULT_SHARE = 0
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

def _share_key(instance_id: str) -> tuple[str, str]:
    return (instance_id, 'vss_share')

def _state(node: Node, layer: int) -> Any:
    return node.get_state(layer)

def _ps(node: Node, layer: int) -> dict:
    return _state(node, layer).protocol_states

def _loop_time() -> float:
    return asyncio.get_running_loop().time()

def _elapsed_ms(t0: float) -> float:
    return (_loop_time() - t0) * 1000.0

def _stage_start() -> tuple[int, int]:
    return (time.process_time_ns(), time.perf_counter_ns())

def _stage_finish(stage_metrics: dict[str, float], stage: str, start: tuple[int, int]) -> float:
    cpu_start, wall_start = start
    cpu_ms = (time.process_time_ns() - cpu_start) / 1000000.0
    wall_ms = (time.perf_counter_ns() - wall_start) / 1000000.0
    stage_metrics[f'{stage}_cpu_ms'] = cpu_ms
    stage_metrics[f'{stage}_wall_ms'] = wall_ms
    stage_metrics[f'{stage}_ms'] = wall_ms
    return wall_ms

def build_disagreement_matrix(ij_sets: dict[int, set[int]], n: int) -> list[list[int]]:
    m = [[0] * n for _ in range(n)]
    for j_rank, bad_parties in ij_sets.items():
        for party in bad_parties:
            i_rank = rank_for_party(party)
            if 0 <= i_rank < n and 0 <= j_rank < n:
                m[i_rank][j_rank] = 1
    return m

def hopcroft_karp(adj: list[list[int]]) -> tuple[dict[int, int], dict[int, int]]:
    n_left = len(adj)
    n_right = max((v for nbrs in adj for v in nbrs), default=-1) + 1
    pair_u = {u: -1 for u in range(n_left)}
    pair_v = {v: -1 for v in range(n_right)}
    dist = {u: 0 for u in range(n_left)}

    def bfs() -> bool:
        queue: list[int] = []
        found = False
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                matched_u = pair_v[v]
                if matched_u == -1:
                    found = True
                elif dist[matched_u] == -1:
                    dist[matched_u] = dist[u] + 1
                    queue.append(matched_u)
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            matched_u = pair_v[v]
            if matched_u == -1 or (dist[matched_u] == dist[u] + 1 and dfs(matched_u)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                dfs(u)
    return (pair_u, pair_v)

def minimum_vertex_cover_from_complaints(M: list[list[int]]) -> tuple[set[int], set[int]]:
    n_left = len(M)
    if n_left == 0:
        return (set(), set())
    n_right = len(M[0])
    for row in M:
        if len(row) != n_right:
            raise ValueError('complaint matrix must be rectangular')
    adj = [[j for j, value in enumerate(row) if value] for row in M]
    pair_u, pair_v = hopcroft_karp(adj)
    z_left: set[int] = set()
    z_right: set[int] = set()
    queue = [u for u in range(n_left) if pair_u[u] == -1]
    z_left.update(queue)
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if pair_u[u] == v or v in z_right:
                continue
            z_right.add(v)
            matched_u = pair_v.get(v, -1)
            if matched_u != -1 and matched_u not in z_left:
                z_left.add(matched_u)
                queue.append(matched_u)
    cover_left = set(range(n_left)) - z_left
    cover_right = z_right
    return (cover_left, cover_right)

def retained_sets_from_complaints(matrix: list[list[int]], n: int, t: int) -> tuple[list[int] | None, list[int] | None, set[int], set[int]]:
    U, V = minimum_vertex_cover_from_complaints(matrix)
    if len(U) + len(V) > 2 * t:
        return (None, None, U, V)
    R = [i for i in range(n) if i not in U]
    C = [j for j in range(n) if j not in V]
    return (R, C, U, V)

def find_rc_sets(matrix: list[list[int]], n: int, t: int) -> tuple[list[int] | None, list[int] | None]:
    R, C, _U, _V = retained_sets_from_complaints(matrix, n, t)
    return (R, C)

def _valid_line(line: object, t: int) -> UniPoly | None:
    if not isinstance(line, UniPoly):
        return None
    if line.degree() > t:
        return None
    return line

def _line_eval_or_default(line: UniPoly | None, x: int) -> int:
    if line is None:
        return DEFAULT_SHARE
    return line.eval(x)

def _raw_line_table(lines: dict[int, UniPoly | None], n: int, k_party: int) -> dict[int, int]:
    return {i_party: _line_eval_or_default(lines.get(i_party), k_party) for i_party in range(1, n + 1)}

def reconstruct_share_at_z(scalar_by_ij: dict[tuple[int, int], int], r_ranks: list[int], c_ranks: list[int], z_party: int, t: int, *, honest: bool=False) -> int:
    _ = honest
    profile_bump(vss_reconstruct_count=1)
    if not r_ranks:
        return DEFAULT_SHARE
    max_j = max((j for _i, j in scalar_by_ij), default=0)
    n = max(max_j, max(c_ranks, default=-1) + 1)
    if n <= 0:
        return DEFAULT_SHARE
    covered_column_count = n - len(c_ranks)

    def _decode_points(points: list[tuple[int, int]], degree: int) -> UniPoly | None:
        profile_bump(bw_decode_count=1, fallback_decode_count=1)
        return berlekamp_welch_decode(points, degree, t)
    row_values: dict[int, int] = {}
    successful_rows: set[int] = set()
    for i_rank in r_ranks:
        i_party = party_id(i_rank)
        points_y = [(j_party, scalar_by_ij.get((i_party, j_party), DEFAULT_SHARE)) for j_party in range(1, n + 1)]
        if len(points_y) < 3 * t + 1:
            return DEFAULT_SHARE
        py = _decode_points(points_y, t)
        if py is not None:
            row_values[i_party] = py.eval(0)
            successful_rows.add(i_party)
        else:
            row_values[i_party] = DEFAULT_SHARE
    if covered_column_count <= t:
        if len(successful_rows) < t + 1:
            return DEFAULT_SHARE
        selected = sorted(successful_rows)[:t + 1]
        return UniPoly.interpolate([(i, row_values[i]) for i in selected]).coeff(0)
    points_x = [(party_id(i_rank), row_values[party_id(i_rank)]) for i_rank in r_ranks]
    if len(points_x) < 3 * t + 1:
        return DEFAULT_SHARE
    gx = _decode_points(points_x, t)
    if gx is None:
        return DEFAULT_SHARE
    return gx.eval(0)

def reconstruct_shares_at_z_batch(scalar_by_batch: list[dict[tuple[int, int], int]], r_ranks: list[int], c_ranks: list[int], z_party: int, t: int, *, honest_flags: list[bool]) -> list[int]:
    batch_size = len(scalar_by_batch)
    profile_bump(vss_reconstruct_count=batch_size)
    if batch_size == 0:
        return []
    profile_bump(com3_scalar_reconstruct_count=batch_size)
    return [reconstruct_share_at_z(scalar_by_batch[bidx], r_ranks, c_ranks, z_party, t, honest=honest_flags[bidx]) for bidx in range(batch_size)]

async def run_vss_dealer(node: Node, start_layer: int, dealer_rank: int, secret: int, instance_id: str, *, rng: Random | None=None) -> float:
    if node.rank != dealer_rank:
        return 0.0
    t0 = _loop_time()
    n = node.n
    t = node.mpc_config.t
    l1 = start_layer + 1
    F = TriPoly.random(t, t, t, constant=mod(secret), rng=rng)
    _ps(node, start_layer)[instance_id] = {'F': F}
    send_tasks: list[asyncio.Task[None]] = []
    for rank in range(n):
        i = party_id(rank)
        slice_i = F.fix_x(i)
        send_tasks.append(asyncio.create_task(node.send_layer_msg(start_layer, l1, rank, PROTOCOL, instance_id, 'slice', {'party': i, 'slice': slice_i})))
    await asyncio.gather(*send_tasks)
    return _elapsed_ms(t0)

async def _com1_forward_slice(node: Node, start_layer: int, instance_id: str, *, byzantine_com1_rank: int | None, rng: Random | None) -> float:
    t0 = _loop_time()
    n = node.n
    t = node.mpc_config.t
    l0, l1, l2 = (start_layer, start_layer + 1, start_layer + 2)
    msgs = await node.recv_until(_msg_filter(instance_id, l1, 'slice', src_layer=l0), 1, 60.0, inbox_key=_inbox_key(instance_id, l1, 'slice', node.rank))
    payload = msgs[0].payload
    assert isinstance(payload, dict)
    slice_i: BiPoly = payload['slice']
    my_party = party_id(node.rank)
    send_tasks: list[asyncio.Task[None]] = []
    for dst_rank in range(n):
        j = party_id(dst_rank)
        if byzantine_com1_rank is not None and node.rank == byzantine_com1_rank:
            line_ij = UniPoly.random(t, rng=rng)
        else:
            line_ij = slice_i.fix_x(j)
        send_tasks.append(asyncio.create_task(node.send_layer_msg(l1, l2, dst_rank, PROTOCOL, instance_id, 'line', {'src_party': my_party, 'dst_party': j, 'line': line_ij})))
    await asyncio.gather(*send_tasks)
    return _elapsed_ms(t0)

async def _com2_process_column(node: Node, start_layer: int, instance_id: str, t: int) -> tuple[set[int], float]:
    t0 = _loop_time()
    n = node.n
    l1, l2 = (start_layer + 1, start_layer + 2)
    my_party = party_id(node.rank)
    msgs = await node.recv_until(_msg_filter(instance_id, l2, 'line', src_layer=l1), n, 60.0, inbox_key=_inbox_key(instance_id, l2, 'line', node.rank))
    raw_lines: dict[int, UniPoly | None] = {i_party: None for i_party in range(1, n + 1)}
    invalid: set[int] = set(range(1, n + 1))
    for m in msgs:
        p = m.payload
        assert isinstance(p, dict)
        src_party = int(p['src_party'])
        line = _valid_line(p.get('line'), t)
        raw_lines[src_party] = line
        if line is not None:
            invalid.discard(src_party)
    _ps(node, l2)[instance_id, 'raw_lines', my_party] = raw_lines
    points = [(i, raw_lines[i] if raw_lines[i] is not None else UniPoly([0])) for i in range(1, n + 1)]
    f_prime = decode_poly_valued(points, t, t, t)
    if f_prime is None:
        return (set(range(1, n + 1)), _elapsed_ms(t0))
    bad: set[int] = set(invalid)
    for i_party, line in raw_lines.items():
        if line is None:
            continue
        expected_line = f_prime.fix_x(i_party)
        if expected_line != line:
            bad.add(i_party)
    _ps(node, l2)[instance_id, 'F_prime', my_party] = f_prime
    return (bad, _elapsed_ms(t0))

async def _com2_send_scalars_and_broadcast_ij(node: Node, start_layer: int, instance_id: str, ij_set: set[int], *, send_values: bool) -> tuple[float, float]:
    _ = send_values
    n = node.n
    l2, l3 = (start_layer + 2, start_layer + 3)
    my_party = party_id(node.rank)
    bcast_start = _stage_start()
    await node.send_layer_msg(l2, l2, 0, PROTOCOL, instance_id, 'ij_set', {'party': my_party, 'bad_parties': sorted(ij_set)}, channel='broadcast')
    broadcast_cpu_ms = (time.process_time_ns() - bcast_start[0]) / 1000000.0
    broadcast_ms = (time.perf_counter_ns() - bcast_start[1]) / 1000000.0
    raw_lines = _ps(node, l2).get((instance_id, 'raw_lines', my_party), {})
    scalar_start = _stage_start()
    send_tasks: list[asyncio.Task[None]] = []
    for dst_rank in range(n):
        k_party = party_id(dst_rank)
        vals = _raw_line_table(raw_lines, n, k_party)
        send_tasks.append(asyncio.create_task(node.send_layer_msg(l2, l3, dst_rank, PROTOCOL, instance_id, 'scalar', {'j_party': my_party, 'k_party': k_party, 'vals': vals})))
    await asyncio.gather(*send_tasks)
    scalar_cpu_ms = (time.process_time_ns() - scalar_start[0]) / 1000000.0
    scalar_ms = (time.perf_counter_ns() - scalar_start[1]) / 1000000.0
    return (broadcast_ms, scalar_ms, broadcast_cpu_ms, scalar_cpu_ms)

async def _collect_ij_sets(node: Node, start_layer: int, instance_id: str) -> dict[int, set[int]]:
    n = node.n
    l2 = start_layer + 2
    msgs = await node.recv_until(_msg_filter(instance_id, l2, 'ij_set', src_layer=l2), n, 60.0, inbox_key=_inbox_key(instance_id, l2, 'ij_set', node.rank))
    ij: dict[int, set[int]] = {}
    for m in msgs:
        p = m.payload
        assert isinstance(p, dict)
        j_party = int(p['party'])
        j_rank = rank_for_party(j_party)
        bad = set((int(x) for x in p['bad_parties']))
        ij[j_rank] = bad
    return ij

async def _com3_reconstruct_share(node: Node, start_layer: int, instance_id: str, r_ranks: list[int] | None, c_ranks: list[int] | None) -> tuple[int, float]:
    t0 = _loop_time()
    t = node.mpc_config.t
    l2, l3 = (start_layer + 2, start_layer + 3)
    k_party = party_id(node.rank)
    if r_ranks is None or c_ranks is None:
        return (DEFAULT_SHARE, _elapsed_ms(t0))
    base_scalar = _msg_filter(instance_id, l3, 'scalar', src_layer=l2)

    def scalar_filter(m: Message) -> bool:
        if not base_scalar(m):
            return False
        if not isinstance(m.payload, dict):
            return False
        try:
            int(m.payload['j_party'])
            return True
        except (KeyError, TypeError, ValueError):
            return False
    msgs = await node.recv_until(scalar_filter, node.n, 60.0)
    scalar_by_ij: dict[tuple[int, int], int] = {}
    for m in msgs:
        p = m.payload
        assert isinstance(p, dict)
        if int(p['k_party']) != k_party:
            continue
        j_party = int(p['j_party'])
        vals = p['vals']
        assert isinstance(vals, dict)
        for i_party, val in vals.items():
            scalar_by_ij[int(i_party), j_party] = mod(int(val))
    honest = bool(_ps(node, start_layer + 2).get(instance_id, {}).get('honest', False))
    share = reconstruct_share_at_z(scalar_by_ij, r_ranks, c_ranks, k_party, t, honest=honest)
    return (share, _elapsed_ms(t0))

async def run_vss_party(node: Node, start_layer: int, dealer_rank: int, instance_id: str, *, byzantine_com1_rank: int | None=None, rng: Random | None=None) -> int:
    n = node.n
    t = node.mpc_config.t
    if n < 4 * t + 1:
        raise ValueError(f'need n >= 4t+1, got n={n}, t={t}')
    l1, l2, l3 = (start_layer + 1, start_layer + 2, start_layer + 3)
    stage_metrics: dict[str, float] = {}
    stage_start = _stage_start()
    await _com1_forward_slice(node, start_layer, instance_id, byzantine_com1_rank=byzantine_com1_rank, rng=rng)
    _stage_finish(stage_metrics, 'com1', stage_start)
    stage_start = _stage_start()
    ij_set, _com2_decode_wall = await _com2_process_column(node, start_layer, instance_id, t)
    _stage_finish(stage_metrics, 'com2_decode', stage_start)
    stage_metrics['com2_broadcast_ms'], stage_metrics['com2_send_scalar_ms'], stage_metrics['com2_broadcast_cpu_ms'], stage_metrics['com2_send_cpu_ms'] = await _com2_send_scalars_and_broadcast_ij(node, start_layer, instance_id, ij_set, send_values=True)
    stage_metrics['com2_broadcast_wall_ms'] = stage_metrics['com2_broadcast_ms']
    stage_metrics['com2_send_wall_ms'] = stage_metrics['com2_send_scalar_ms']
    stage_start = _stage_start()
    all_ij = await _collect_ij_sets(node, start_layer, instance_id)
    for j_rank in range(n):
        all_ij.setdefault(j_rank, set(range(1, n + 1)))
    matrix = build_disagreement_matrix(all_ij, n)
    r_ranks, c_ranks, u_cover, v_cover = retained_sets_from_complaints(matrix, n, t)
    _stage_finish(stage_metrics, 'public', stage_start)
    _ps(node, l2)[instance_id] = {'R': r_ranks, 'C': c_ranks, 'U': sorted(u_cover), 'V': sorted(v_cover), 'matrix': matrix, 'honest': False}
    stage_start = _stage_start()
    share, _com3_wall = await _com3_reconstruct_share(node, start_layer, instance_id, r_ranks, c_ranks)
    _stage_finish(stage_metrics, 'com3', stage_start)
    _ps(node, start_layer).setdefault(instance_id, {})['stage_metrics'] = stage_metrics
    _state(node, l3).values[_share_key(instance_id)] = share
    return share

async def run_vss(node: Node, start_layer: int, dealer_rank: int, secret: int, instance_id: str, *, byzantine_com1_rank: int | None=None, rng: Random | None=None) -> int:
    dealer_start = _stage_start()
    dealer_ms = await run_vss_dealer(node, start_layer, dealer_rank, secret, instance_id, rng=rng)
    dealer_cpu_ms = (time.process_time_ns() - dealer_start[0]) / 1000000.0
    dealer_wall_ms = (time.perf_counter_ns() - dealer_start[1]) / 1000000.0
    share = await run_vss_party(node, start_layer, dealer_rank, instance_id, byzantine_com1_rank=byzantine_com1_rank, rng=rng)
    party_stages = _ps(node, start_layer).get(instance_id, {}).get('stage_metrics', {})
    party_stages['dealer_ms'] = dealer_ms
    party_stages['dealer_cpu_ms'] = dealer_cpu_ms
    party_stages['dealer_wall_ms'] = dealer_wall_ms
    _ps(node, start_layer).setdefault(instance_id, {})['stage_metrics'] = party_stages
    return share

async def run_vss_dealer_batch(node: Node, start_layer: int, dealer_rank: int, secrets: list[int], instance_id: str, *, rng: Random | None=None) -> float:
    if node.rank != dealer_rank:
        return 0.0
    t0 = _loop_time()
    n = node.n
    t = node.mpc_config.t
    l1 = start_layer + 1
    batch_size = len(secrets)
    polys: list[TriPoly] = []
    for idx, secret in enumerate(secrets):
        polys.append(TriPoly.random(t, t, t, constant=mod(secret), rng=rng))
    _ps(node, start_layer)[instance_id] = {'F_batch': polys, 'batch_size': batch_size}
    send_tasks: list[asyncio.Task[None]] = []
    for rank in range(n):
        i = party_id(rank)
        slices = [poly.fix_x(i) for poly in polys]
        send_tasks.append(asyncio.create_task(node.send_layer_msg(start_layer, l1, rank, PROTOCOL, instance_id, 'slice_batch', {'party': i, 'slices': slices})))
    await asyncio.gather(*send_tasks)
    return _elapsed_ms(t0)

async def _com1_forward_slice_batch(node: Node, start_layer: int, instance_id: str, batch_size: int, *, byzantine_com1_rank: int | None, rng: Random | None) -> float:
    t0 = _loop_time()
    n = node.n
    t = node.mpc_config.t
    l0, l1, l2 = (start_layer, start_layer + 1, start_layer + 2)
    msgs = await node.recv_until(_msg_filter(instance_id, l1, 'slice_batch', src_layer=l0), 1, 60.0, inbox_key=_inbox_key(instance_id, l1, 'slice_batch', node.rank))
    payload = msgs[0].payload
    assert isinstance(payload, dict)
    slices: list[BiPoly] = payload['slices']
    if len(slices) != batch_size:
        raise ValueError(f'expected {batch_size} slices, got {len(slices)}')
    my_party = party_id(node.rank)
    send_tasks: list[asyncio.Task[None]] = []
    for dst_rank in range(n):
        j = party_id(dst_rank)
        if byzantine_com1_rank is not None and node.rank == byzantine_com1_rank:
            lines = [UniPoly.random(t, rng=rng) for _ in range(batch_size)]
        else:
            lines = [slice_i.fix_x(j) for slice_i in slices]
        send_tasks.append(asyncio.create_task(node.send_layer_msg(l1, l2, dst_rank, PROTOCOL, instance_id, 'line_batch', {'src_party': my_party, 'dst_party': j, 'lines': lines})))
    await asyncio.gather(*send_tasks)
    return _elapsed_ms(t0)

async def _com2_process_column_batch(node: Node, start_layer: int, instance_id: str, batch_size: int, t: int) -> tuple[list[set[int]], float]:
    t0 = _loop_time()
    n = node.n
    l1, l2 = (start_layer + 1, start_layer + 2)
    my_party = party_id(node.rank)
    profiler = get_profiler()
    with profiler.op('masked_value_collect', count=n):
        msgs = await node.recv_until(_msg_filter(instance_id, l2, 'line_batch', src_layer=l1), n, 60.0, inbox_key=_inbox_key(instance_id, l2, 'line_batch', node.rank))
        lines_by_src: dict[int, list[object]] = {}
        for m in msgs:
            p = m.payload
            assert isinstance(p, dict)
            src_party = int(p['src_party'])
            lines_by_src[src_party] = p['lines'] if isinstance(p.get('lines'), list) else []
    lines_by_batch: list[dict[int, UniPoly | None]] = []
    for batch_idx in range(batch_size):
        lines: dict[int, UniPoly | None] = {i_party: None for i_party in range(1, n + 1)}
        for src_party in range(1, n + 1):
            line_list = lines_by_src.get(src_party, [])
            if batch_idx < len(line_list):
                lines[src_party] = _valid_line(line_list[batch_idx], t)
        lines_by_batch.append(lines)
    bad_sets: list[set[int]] = []
    decode_inputs: list[dict[int, UniPoly]] = []
    invalid_by_batch: list[set[int]] = []
    for lines in lines_by_batch:
        invalid = {i_party for i_party, line in lines.items() if line is None}
        invalid_by_batch.append(invalid)
        decode_inputs.append({i_party: line if line is not None else UniPoly([0]) for i_party, line in lines.items()})
    with profiler.op('fast_interpolate', count=batch_size):
        f_primes = decode_poly_valued_batch(decode_inputs, t, t, t, fast_path=True)
    with profiler.op('local_check', count=batch_size * n):
        for batch_idx, f_prime in enumerate(f_primes):
            _ps(node, l2)[instance_id, 'raw_lines', my_party, batch_idx] = lines_by_batch[batch_idx]
            if f_prime is None:
                bad_sets.append(set(range(1, n + 1)))
                _ps(node, l2)[instance_id, 'F_prime', my_party, batch_idx] = None
                continue
            lines = lines_by_batch[batch_idx]
            bad: set[int] = set(invalid_by_batch[batch_idx])
            for i_party, line in lines.items():
                if line is None:
                    continue
                expected_line = f_prime.fix_x(i_party)
                if expected_line != line:
                    bad.add(i_party)
            bad_sets.append(bad)
            _ps(node, l2)[instance_id, 'F_prime', my_party, batch_idx] = f_prime
    return (bad_sets, _elapsed_ms(t0))

async def _com2_send_scalars_batch(node: Node, start_layer: int, instance_id: str, ij_sets: list[set[int]], batch_size: int, *, send_values: bool) -> tuple[float, float, float, float]:
    _ = send_values
    n = node.n
    l2, l3 = (start_layer + 2, start_layer + 3)
    my_party = party_id(node.rank)
    profiler = get_profiler()
    profiler.stage = 'vss_com2_send'
    bcast_start = _stage_start()
    with profiler.op('ij_set_payload_build', count=1):
        profile_bump(broadcast_payload_count=1, broadcast_scalar_count=batch_size * n)
        await node.send_layer_msg(l2, l2, 0, PROTOCOL, instance_id, 'ij_set_batch', {'kind': 'ij_set_batch', 'party': my_party, 'batch_size': batch_size, 'bad_parties_list': [sorted(bad) for bad in ij_sets]}, channel='broadcast')
    broadcast_cpu_ms = (time.process_time_ns() - bcast_start[0]) / 1000000.0
    broadcast_ms = (time.perf_counter_ns() - bcast_start[1]) / 1000000.0
    scalar_start = _stage_start()
    raw_batches = [_ps(node, l2).get((instance_id, 'raw_lines', my_party, batch_idx), {}) for batch_idx in range(batch_size)]
    send_tasks: list[asyncio.Task[None]] = []
    with profiler.op('scalar_table_build', count=n):
        for dst_rank in range(n):
            k_party = party_id(dst_rank)
            table = [[0] * batch_size for _ in range(n)]
            for batch_idx, raw_lines in enumerate(raw_batches):
                for i_idx in range(n):
                    table[i_idx][batch_idx] = _line_eval_or_default(raw_lines.get(i_idx + 1), k_party)
            payload = scalar_table_payload(j_party=my_party, k_party=k_party, batch_size=batch_size, table=table)
            with profiler.op('message_pack', count=1):
                profile_bump(vss_send_scalar_count=1)
                send_tasks.append(asyncio.create_task(node.send_layer_msg(l2, l3, dst_rank, PROTOCOL, instance_id, 'scalar_batch', payload)))
    with profiler.op('receiver_loop', count=len(send_tasks)):
        await asyncio.gather(*send_tasks)
    scalar_cpu_ms = (time.process_time_ns() - scalar_start[0]) / 1000000.0
    scalar_ms = (time.perf_counter_ns() - scalar_start[1]) / 1000000.0
    return (broadcast_ms, scalar_ms, broadcast_cpu_ms, scalar_cpu_ms)

async def _collect_ij_sets_batch(node: Node, start_layer: int, instance_id: str, batch_size: int) -> dict[int, list[set[int]]]:
    n = node.n
    l2 = start_layer + 2
    msgs = await node.recv_until(_msg_filter(instance_id, l2, 'ij_set_batch', src_layer=l2), n, 60.0, inbox_key=_inbox_key(instance_id, l2, 'ij_set_batch', node.rank))
    ij: dict[int, list[set[int]]] = {}
    for m in msgs:
        p = m.payload
        assert isinstance(p, dict)
        j_party = int(p['party'])
        j_rank = rank_for_party(j_party)
        bad_list = p['bad_parties_list']
        ij[j_rank] = [set((int(x) for x in bad)) for bad in bad_list]
        if len(ij[j_rank]) != batch_size:
            raise ValueError(f'expected {batch_size} ij sets from party {j_party}, got {len(ij[j_rank])}')
    return ij

async def _com3_reconstruct_shares_batch(node: Node, start_layer: int, instance_id: str, r_c_by_batch: list[tuple[list[int] | None, list[int] | None]], batch_size: int) -> tuple[list[int], float]:
    t0 = _loop_time()
    t = node.mpc_config.t
    l2, l3 = (start_layer + 2, start_layer + 3)
    k_party = party_id(node.rank)
    shares: list[int] = [0] * batch_size
    needed_batches = [idx for idx in range(batch_size) if r_c_by_batch[idx][0] is not None and r_c_by_batch[idx][1] is not None]
    if not needed_batches:
        return ([DEFAULT_SHARE] * batch_size, _elapsed_ms(t0))
    base_scalar = _msg_filter(instance_id, l3, 'scalar_batch', src_layer=l2)

    def scalar_filter(m: Message) -> bool:
        if not base_scalar(m):
            return False
        if not isinstance(m.payload, dict):
            return False
        try:
            int(m.payload['j_party'])
            return True
        except (KeyError, TypeError, ValueError):
            return False
    msgs = await node.recv_until(scalar_filter, node.n, 60.0)
    scalar_by_batch: list[dict[tuple[int, int], int]] = [{} for _ in range(batch_size)]
    profiler = get_profiler()
    with profiler.op('share_collect', count=len(msgs)):
        for m in msgs:
            p = m.payload
            assert isinstance(p, dict)
            j_party, recv_k, recv_bs, table = parse_scalar_table(p)
            if recv_k != k_party:
                continue
            for i_idx, row in enumerate(table):
                i_party = i_idx + 1
                for batch_idx, val in enumerate(row):
                    if batch_idx >= batch_size:
                        break
                    scalar_by_batch[batch_idx][i_party, j_party] = mod(int(val))
    honest_flags = [False for _ in range(batch_size)]
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for batch_idx in needed_batches:
        r_ranks, c_ranks = r_c_by_batch[batch_idx]
        assert r_ranks is not None and c_ranks is not None
        key = (tuple(r_ranks), tuple(c_ranks))
        groups.setdefault(key, []).append(batch_idx)
    for (r_key, c_key), indices in groups.items():
        r_ranks = list(r_key)
        c_ranks = list(c_key)
        sub_tables = [scalar_by_batch[i] for i in indices]
        sub_honest = [honest_flags[i] for i in indices]
        sub_shares = reconstruct_shares_at_z_batch(sub_tables, r_ranks, c_ranks, k_party, t, honest_flags=sub_honest)
        for batch_idx, share in zip(indices, sub_shares, strict=True):
            shares[batch_idx] = share
    return (shares, _elapsed_ms(t0))

async def _com3_reconstruct_share_batch(node: Node, start_layer: int, instance_id: str, r_ranks: list[int] | None, c_ranks: list[int] | None, batch_size: int) -> tuple[list[int], float]:
    r_c_by_batch = [(r_ranks, c_ranks)] * batch_size
    return await _com3_reconstruct_shares_batch(node, start_layer, instance_id, r_c_by_batch, batch_size)

async def run_vss_party_batch(node: Node, start_layer: int, dealer_rank: int, instance_id: str, batch_size: int, *, byzantine_com1_rank: int | None=None, rng: Random | None=None) -> list[int]:
    _ = dealer_rank
    n = node.n
    t = node.mpc_config.t
    if n < 4 * t + 1:
        raise ValueError(f'need n >= 4t+1, got n={n}, t={t}')
    l2, l3 = (start_layer + 2, start_layer + 3)
    stage_metrics: dict[str, float] = {}
    profiler = get_profiler()
    stage_start = _stage_start()
    profiler.stage = 'vss_com1'
    await _com1_forward_slice_batch(node, start_layer, instance_id, batch_size, byzantine_com1_rank=byzantine_com1_rank, rng=rng)
    _stage_finish(stage_metrics, 'com1', stage_start)
    stage_start = _stage_start()
    profiler.stage = 'vss_com2_decode'
    ij_sets, _com2_decode_wall = await _com2_process_column_batch(node, start_layer, instance_id, batch_size, t)
    _stage_finish(stage_metrics, 'com2_decode', stage_start)
    profiler.stage = 'vss_com2_broadcast'
    stage_metrics['com2_broadcast_ms'], stage_metrics['com2_send_scalar_ms'], stage_metrics['com2_broadcast_cpu_ms'], stage_metrics['com2_send_cpu_ms'] = await _com2_send_scalars_batch(node, start_layer, instance_id, ij_sets, batch_size, send_values=True)
    stage_metrics['com2_broadcast_wall_ms'] = stage_metrics['com2_broadcast_ms']
    stage_metrics['com2_send_wall_ms'] = stage_metrics['com2_send_scalar_ms']
    stage_start = _stage_start()
    profiler.stage = 'vss_public'
    all_ij = await _collect_ij_sets_batch(node, start_layer, instance_id, batch_size)
    r_c_by_batch: list[tuple[list[int] | None, list[int] | None]] = []
    honest_by_batch: list[bool] = []
    for batch_idx in range(batch_size):
        ij_for_idx = {j_rank: bad_list[batch_idx] for j_rank, bad_list in all_ij.items()}
        for j_rank in range(n):
            ij_for_idx.setdefault(j_rank, set(range(1, n + 1)))
        matrix = build_disagreement_matrix(ij_for_idx, n)
        r_ranks, c_ranks, u_cover, v_cover = retained_sets_from_complaints(matrix, n, t)
        honest_by_batch.append(False)
        r_c_by_batch.append((r_ranks, c_ranks))
        _ps(node, l2)[instance_id, batch_idx, 'cover'] = {'U': sorted(u_cover), 'V': sorted(v_cover), 'matrix': matrix}
    _stage_finish(stage_metrics, 'public', stage_start)
    for batch_idx, (r_ranks, c_ranks) in enumerate(r_c_by_batch):
        _ps(node, l2)[instance_id, batch_idx] = {'R': r_ranks, 'C': c_ranks, 'honest': honest_by_batch[batch_idx]}
    stage_start = _stage_start()
    profiler.stage = 'vss_com3'
    shares, _com3_wall = await _com3_reconstruct_shares_batch(node, start_layer, instance_id, r_c_by_batch, batch_size)
    _stage_finish(stage_metrics, 'com3', stage_start)
    _ps(node, start_layer).setdefault(instance_id, {})['stage_metrics'] = stage_metrics
    for batch_idx, share in enumerate(shares):
        _state(node, l3).values[instance_id, 'vss_share', batch_idx] = share
    return shares

async def run_vss_batch(node: Node, start_layer: int, dealer_rank: int, secrets: list[int], instance_id: str, *, byzantine_com1_rank: int | None=None, rng: Random | None=None) -> list[int]:
    batch_size = len(secrets)
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    dealer_start = _stage_start()
    dealer_ms = await run_vss_dealer_batch(node, start_layer, dealer_rank, secrets, instance_id, rng=rng)
    dealer_cpu_ms = (time.process_time_ns() - dealer_start[0]) / 1000000.0
    dealer_wall_ms = (time.perf_counter_ns() - dealer_start[1]) / 1000000.0
    shares = await run_vss_party_batch(node, start_layer, dealer_rank, instance_id, batch_size, byzantine_com1_rank=byzantine_com1_rank, rng=rng)
    party_stages = _ps(node, start_layer).get(instance_id, {}).get('stage_metrics', {})
    party_stages['dealer_ms'] = dealer_ms
    party_stages['dealer_cpu_ms'] = dealer_cpu_ms
    party_stages['dealer_wall_ms'] = dealer_wall_ms
    _ps(node, start_layer).setdefault(instance_id, {})['stage_metrics'] = party_stages
    return shares

async def run_discard_vss_warmup(node: Node, start_layer: int, instance_id: str, rng: Random | None=None, *, dealer_rank: int=0, layer_offset: int=1000) -> None:
    warmup_layer = start_layer + layer_offset
    warmup_id = f'{instance_id}:warmup'
    secret = rand_field(rng) if node.rank == dealer_rank else 0
    await run_vss(node=node, start_layer=warmup_layer, dealer_rank=dealer_rank, secret=secret, instance_id=warmup_id, rng=rng)
    for layer in range(warmup_layer, warmup_layer + 4):
        node.clear_state(layer)

@dataclass
class VSSNodeResult:
    rank: int
    pid: int
    share: int
    metrics: dict[str, int]
    protocol_time_ms: float = 0.0
    stage_metrics: dict[str, float] | None = None
STAGE_METRIC_KEYS = ('dealer_ms', 'com1_ms', 'com2_decode_ms', 'com2_broadcast_ms', 'com2_send_scalar_ms', 'public_ms', 'com3_ms')

def aggregate_stage_metrics(results: list[VSSNodeResult], *, dealer_rank: int=0) -> dict[str, float]:
    if not results:
        return {k: 0.0 for k in STAGE_METRIC_KEYS}

    def _max_key(key: str) -> float:
        vals = [(r.stage_metrics or {}).get(key, 0.0) for r in results]
        return max(vals) if vals else 0.0
    dealer_ms = 0.0
    for r in results:
        if r.rank == dealer_rank:
            dealer_ms = (r.stage_metrics or {}).get('dealer_ms', 0.0)
            break
    if dealer_ms == 0.0:
        dealer_ms = _max_key('dealer_ms')
    return {'dealer_ms': dealer_ms, 'com1_ms': _max_key('com1_ms'), 'com2_decode_ms': _max_key('com2_decode_ms'), 'com2_broadcast_ms': _max_key('com2_broadcast_ms'), 'com2_send_scalar_ms': _max_key('com2_send_scalar_ms'), 'public_ms': _max_key('public_ms'), 'com3_ms': _max_key('com3_ms')}

async def _run_vss_node_async(rank: int, n: int, t: int | None, base_port: int, start_layer: int, dealer_rank: int, secret: int, instance_id: str, startup_delay: float, byzantine_com1_rank: int | None, warmup_round: bool, hosts: tuple[str, ...] | None=None) -> VSSNodeResult:
    import os
    from cubempc.config import MPCConfig
    from cubempc.network.node import NodeProcess
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    node = NodeProcess(rank, n, base_port, mpc)
    await node.start_server()
    await asyncio.sleep(startup_delay)
    await node.connect_peers()
    rng = Random(rank + 17)
    if warmup_round:
        await run_discard_vss_warmup(node, start_layer, instance_id, rng, dealer_rank=dealer_rank)
    t0 = asyncio.get_running_loop().time()
    share = await run_vss(node, start_layer, dealer_rank, secret, instance_id, byzantine_com1_rank=byzantine_com1_rank, rng=rng)
    protocol_time_ms = (asyncio.get_running_loop().time() - t0) * 1000.0
    await node.stop_server()
    from cubempc.network.coordinator import _metrics_to_dict
    stage_metrics = _ps(node, start_layer).get(instance_id, {}).get('stage_metrics')
    if stage_metrics is not None:
        stage_metrics = dict(stage_metrics)
    return VSSNodeResult(rank=rank, pid=os.getpid(), share=share, metrics=_metrics_to_dict(node.metrics), protocol_time_ms=protocol_time_ms, stage_metrics=stage_metrics)

def run_vss_multiprocess(n: int, secret: int, *, t: int | None=None, base_port: int=19000, start_layer: int=0, dealer_rank: int=0, instance_id: str='vss-test', startup_delay: float=2.0, byzantine_com1_rank: int | None=None, warmup_round: bool=False, timeout: float=120.0, hosts: tuple[str, ...] | None=None) -> dict[str, Any]:
    import multiprocessing as mp
    import time
    from queue import Empty
    ctx = mp.get_context('spawn')
    out_queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []
    for rank in range(n):
        proc = ctx.Process(target=_vss_worker_entry, args=(rank, n, t, base_port, start_layer, dealer_rank, secret, instance_id, startup_delay, byzantine_com1_rank, warmup_round, hosts, out_queue), name=f'vss-{rank}')
        proc.start()
        processes.append(proc)
    deadline = time.time() + timeout
    results: list[VSSNodeResult] = []
    while len(results) < n and time.time() < deadline:
        try:
            item = out_queue.get(timeout=0.5)
            if isinstance(item, VSSNodeResult):
                results.append(item)
            elif isinstance(item, dict) and 'error' in item:
                raise RuntimeError(f'VSS node {item.get('rank')} failed: {item['error']}')
        except Empty:
            continue
    from cubempc.network.coordinator import join_worker_processes
    join_worker_processes(processes)
    if len(results) < n:
        raise RuntimeError(f'expected {n} VSS results, got {len(results)}')
    results.sort(key=lambda r: r.rank)
    shares = {party_id(r.rank): r.share for r in results}
    from cubempc.config import MPCConfig
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    reconstructed = robust_reconstruct(shares, mpc.t, mpc.t)
    agg = __import__('cubempc.metrics', fromlist=['Metrics']).Metrics()
    from cubempc.network.coordinator import _metrics_from_dict
    for r in results:
        agg.merge(_metrics_from_dict(r.metrics))
    from cubempc.network.coordinator import _metrics_to_dict
    total_time_ms = max((r.protocol_time_ms for r in results)) if results else 0.0
    return {'n': n, 't': mpc.t, 'secret': mod(secret), 'reconstructed': reconstructed, 'shares': shares, 'per_rank': [r.__dict__ for r in results], 'aggregated_metrics': _metrics_to_dict(agg), 'total_time_ms': total_time_ms}

def _vss_worker_entry(*args: object, **kwargs: object) -> None:
    import asyncio
    import os
    import traceback
    from queue import Queue
    rank, n, t, base_port, start_layer, dealer_rank, secret, instance_id, startup_delay, byzantine_com1_rank, warmup_round, hosts, out_queue = args
    q: Queue = out_queue
    try:
        result = asyncio.run(_run_vss_node_async(int(rank), int(n), t, int(base_port), int(start_layer), int(dealer_rank), int(secret), str(instance_id), float(startup_delay), byzantine_com1_rank if byzantine_com1_rank is None else int(byzantine_com1_rank), bool(warmup_round), hosts if hosts is None else tuple(hosts)))
        q.put(result)
    except Exception as exc:
        q.put({'rank': rank, 'error': traceback.format_exc(), 'pid': os.getpid()})
run_layered_vss = run_vss_multiprocess