from __future__ import annotations
import argparse
import asyncio
import csv
import hashlib
import multiprocessing as mp
import pickle
import socket
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from random import Random
from typing import Any
from cubempc.benchmarks.distributed import distributed_layout, launch_remote_rank, load_worker_result, wait_for_result_files
from cubempc.config import MPCConfig, default_threshold, validate_party_count
from cubempc.field import mod, rand_field
from cubempc.metrics import FIELD_ELEMENT_BYTES, rg_logical_cost
from cubempc.network.coordinator import join_worker_processes
from cubempc.network.node import NodeProcess
from cubempc.host_layout import HostLayout
from cubempc.protocols.multiplication import resharing_prep_keys, run_mul
from cubempc.protocols.rg import map_rg_output_to_mul_keys, map_rg_output_to_rs_keys, run_rg
from cubempc.protocols.vss import party_id, run_discard_vss_warmup
from cubempc.shamir import robust_reconstruct, shamir_share
PROTOCOL = 'MUL'
LOGICAL_RAW_FIELDS = ['prep_logical_p2p_bytes', 'prep_logical_broadcast_bytes', 'prep_logical_total_bytes', 'prep_logical_message_count', 'online_logical_p2p_bytes', 'online_logical_broadcast_bytes', 'online_logical_total_bytes', 'online_logical_message_count', 'logical_p2p_bytes', 'logical_broadcast_bytes', 'logical_total_bytes', 'logical_message_count']
RAW_FIELDS = ['protocol', 'n', 't', 'q', 'repeat_id', 'prep_time_ms', 'online_time_ms', 'total_time_ms', *LOGICAL_RAW_FIELDS, 'success', 'error']
SUMMARY_FIELDS = ['protocol', 'n', 't', 'q', 'success_count', 'total_count', 'success_rate', 'prep_time_ms_mean', 'prep_time_ms_std', 'online_time_ms_mean', 'online_time_ms_std', 'total_time_ms_mean', 'total_time_ms_std', *(f'{field}_mean' for field in LOGICAL_RAW_FIELDS)]
PORT_STRIDE = 1000

def _mul_prep_logical_cost(n: int, t: int) -> dict[str, int]:
    rg = rg_logical_cost(n, t)
    mask_eval_bytes = 2 * n * n * FIELD_ELEMENT_BYTES
    mask_eval_messages = n * n
    return {'logical_p2p_bytes': rg['logical_p2p_bytes'] + mask_eval_bytes, 'logical_broadcast_bytes': rg['logical_broadcast_bytes'], 'logical_total_bytes': rg['logical_total_bytes'] + mask_eval_bytes, 'logical_message_count': rg['logical_message_count'] + mask_eval_messages}

def _mul_online_logical_cost(n: int, _t: int) -> dict[str, int]:
    rs_bytes = n * n * FIELD_ELEMENT_BYTES
    broadcast_bytes = n * FIELD_ELEMENT_BYTES
    return {'logical_p2p_bytes': rs_bytes, 'logical_broadcast_bytes': broadcast_bytes, 'logical_total_bytes': rs_bytes + broadcast_bytes, 'logical_message_count': n + n * n}

def _logical_columns(prep: dict[str, int], online: dict[str, int]) -> dict[str, int]:
    return {'prep_logical_p2p_bytes': prep['logical_p2p_bytes'], 'prep_logical_broadcast_bytes': prep['logical_broadcast_bytes'], 'prep_logical_total_bytes': prep['logical_total_bytes'], 'prep_logical_message_count': prep['logical_message_count'], 'online_logical_p2p_bytes': online['logical_p2p_bytes'], 'online_logical_broadcast_bytes': online['logical_broadcast_bytes'], 'online_logical_total_bytes': online['logical_total_bytes'], 'online_logical_message_count': online['logical_message_count'], 'logical_p2p_bytes': prep['logical_p2p_bytes'] + online['logical_p2p_bytes'], 'logical_broadcast_bytes': prep['logical_broadcast_bytes'] + online['logical_broadcast_bytes'], 'logical_total_bytes': prep['logical_total_bytes'] + online['logical_total_bytes'], 'logical_message_count': prep['logical_message_count'] + online['logical_message_count']}

def _port_block_available(base_port: int, n: int) -> bool:
    sockets: list[socket.socket] = []
    try:
        for port in range(base_port, base_port + n):
            sock = socket.socket()
            sock.bind(('127.0.0.1', port))
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()

def _next_free_base_port(candidate: int, n: int, used_bases: set[int]) -> int:
    base = candidate
    for _ in range(100):
        if base + n < 65535 and base not in used_bases and _port_block_available(base, n):
            return base
        base += PORT_STRIDE
    for _ in range(100):
        probe = socket.socket()
        probe.bind(('127.0.0.1', 0))
        base = probe.getsockname()[1]
        probe.close()
        if base + n < 65535 and base not in used_bases and _port_block_available(base, n):
            return base
    raise RuntimeError(f'could not find a free port block starting at {candidate}')

@dataclass
class MulStandaloneNodeResult:
    rank: int
    pid: int
    product_share: int
    rg_prep_ms: float
    com0_mask_ms: float
    online_ms: float

async def _run_rg_prep_on_node(node: NodeProcess, *, q: int, start_layer: int, instance_id: str, rng: Random, warmup_round: bool) -> tuple[list[int], float]:
    loop = asyncio.get_running_loop()
    if warmup_round:
        await run_discard_vss_warmup(node, start_layer, instance_id, rng)
    t0 = loop.time()
    shares = await run_rg(node, start_layer, q, instance_id, rng=rng, warmup_round=False)
    prep_ms = (loop.time() - t0) * 1000.0
    return (shares, prep_ms)

async def _run_mul_node_async(rank: int, n: int, t: int | None, base_port: int, layer: int, x: int, y: int, instance_id: str, startup_delay: float, recv_timeout: float, warmup_round: bool, hosts: tuple[str, ...] | None=None) -> MulStandaloneNodeResult:
    import os
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    node = NodeProcess(rank, n, base_port, mpc)
    await node.start_server()
    await asyncio.sleep(startup_delay)
    await node.connect_peers()
    setup_rng = Random(int.from_bytes(hashlib.sha256(f'{instance_id}:setup'.encode()).digest()[:8], 'big'))
    state = node.get_state(layer)
    x_key = (instance_id, 'x')
    y_key = (instance_id, 'y')
    output_key = (instance_id, 'xy_out')
    state.values[x_key] = shamir_share(mod(x), n, mpc.t, rng=setup_rng)[party_id(rank)]
    state.values[y_key] = shamir_share(mod(y), n, mpc.t, rng=setup_rng)[party_id(rank)]
    rg_rng = Random(int.from_bytes(hashlib.sha256(f'{instance_id}:rg:{rank}'.encode()).digest()[:8], 'big'))
    rg_id = f'{instance_id}:rg-for-mul'
    q = 3 * mpc.t + 1
    rg_shares, rg_prep_ms = await _run_rg_prep_on_node(node, q=q, start_layer=layer, instance_id=rg_id, rng=rg_rng, warmup_round=warmup_round)
    random_keys = map_rg_output_to_mul_keys(node, layer, instance_id, rg_shares, mpc.t)
    adapter_rs_keys = map_rg_output_to_rs_keys(node, layer, f'{instance_id}:mul-internal-rs', rg_shares, mpc.t)
    for src_key, dst_key in zip(adapter_rs_keys, resharing_prep_keys(instance_id, mpc.t), strict=True):
        state.values[dst_key] = state.values[src_key]
    timing: dict[str, float] = {}
    loop = asyncio.get_running_loop()
    fallback_t0 = loop.time()
    product_share = await run_mul(node=node, layer=layer, x_key=x_key, y_key=y_key, random_keys=random_keys, output_key=output_key, instance_id=instance_id, recv_timeout=recv_timeout, timing=timing)
    fallback_elapsed_ms = (loop.time() - fallback_t0) * 1000.0
    com0_mask_ms = timing.get('com0_mask_ms', 0.0)
    online_ms = timing.get('online_ms', fallback_elapsed_ms)
    await node.stop_server()
    return MulStandaloneNodeResult(rank=rank, pid=os.getpid(), product_share=product_share, rg_prep_ms=rg_prep_ms, com0_mask_ms=com0_mask_ms, online_ms=online_ms)

def _worker_entry(*args: object) -> None:
    import traceback
    from queue import Queue
    rank, n, t, base_port, layer, x, y, instance_id, startup_delay, recv_timeout, warmup_round, hosts, out_queue = args
    q_out: Queue = out_queue
    try:
        result = asyncio.run(_run_mul_node_async(int(rank), int(n), t, int(base_port), int(layer), int(x), int(y), str(instance_id), float(startup_delay), float(recv_timeout), bool(warmup_round), hosts if hosts is None else tuple(hosts)))
        q_out.put(result)
    except Exception:
        q_out.put({'rank': rank, 'error': traceback.format_exc()})

def run_mul_standalone_multiprocess(n: int, *, t: int | None, base_port: int, x: int, y: int, instance_id: str, startup_delay: float, timeout: float, recv_timeout: float, warmup_round: bool=False, hosts: tuple[str, ...] | None=None, remote_repo: str | None=None, ssh_user: str | None=None, shared_tmp_dir: Path | None=None) -> dict[str, Any]:
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    results: list[MulStandaloneNodeResult] = []
    first_error: str | None = None
    layout = mpc.host_layout()
    if distributed_layout(layout):
        if not remote_repo:
            raise ValueError('--remote-repo-dir is required when --hosts spans multiple machines')
        if shared_tmp_dir is None:
            raise ValueError('--shared-tmp-dir is required when --hosts spans multiple machines')
        with tempfile.TemporaryDirectory(prefix='mul-standalone-') as tmp:
            _ = tmp
            work_dir = shared_tmp_dir / instance_id
            work_dir.mkdir(parents=True, exist_ok=True)
            procs: list[Any] = []
            result_paths: list[Path] = []
            for rank in range(n):
                params_path = work_dir / f'rank{rank}-params.pkl'
                result_path = work_dir / f'rank{rank}-result.pkl'
                params = {'module': __name__, 'function': '_run_mul_node_async', 'kwargs': {'rank': rank, 'n': n, 't': t, 'base_port': base_port, 'layer': 0, 'x': x, 'y': y, 'instance_id': instance_id, 'startup_delay': startup_delay, 'recv_timeout': recv_timeout, 'warmup_round': warmup_round, 'hosts': hosts}}
                with params_path.open('wb') as fh:
                    pickle.dump(params, fh)
                procs.append(launch_remote_rank(rank=rank, host=layout.host_for_rank(rank), remote_repo=remote_repo, python_executable=sys.executable, worker_args=['--params-file', str(params_path), '--result-file', str(result_path)], ssh_user=ssh_user, worker_module='cubempc.benchmarks.subprotocol_worker'))
                result_paths.append(result_path)
            wait_for_result_files(result_paths, timeout=timeout)
            for proc in procs:
                rc = proc.wait(timeout=60)
                if rc != 0 and first_error is None:
                    first_error = f'remote MUL worker exit code {rc}'
            for result_path in result_paths:
                item = load_worker_result(result_path)
                if isinstance(item, MulStandaloneNodeResult):
                    results.append(item)
                elif isinstance(item, dict) and 'error' in item:
                    first_error = f'MUL node {item.get('rank')} failed: {item['error']}'
                    break
    else:
        ctx = mp.get_context('spawn')
        out_queue: mp.Queue = ctx.Queue()
        processes: list[mp.Process] = []
        for rank in range(n):
            proc = ctx.Process(target=_worker_entry, args=(rank, n, t, base_port, 0, x, y, instance_id, startup_delay, recv_timeout, warmup_round, hosts, out_queue), name=f'mul-standalone-{rank}')
            proc.start()
            processes.append(proc)
        deadline = time.time() + timeout
        while len(results) < n and time.time() < deadline:
            try:
                item = out_queue.get(timeout=0.5)
                if isinstance(item, MulStandaloneNodeResult):
                    results.append(item)
                elif isinstance(item, dict) and 'error' in item:
                    first_error = f'MUL node {item.get('rank')} failed: {item['error']}'
                    break
            except Empty:
                continue
        join_worker_processes(processes)
    if first_error is not None:
        raise RuntimeError(first_error)
    if len(results) < n:
        raise RuntimeError(f'expected {n} MUL results, got {len(results)}')
    results.sort(key=lambda r: r.rank)
    rg_prep_ms = max((r.rg_prep_ms for r in results))
    com0_ms = max((r.com0_mask_ms for r in results))
    prep_time_ms = rg_prep_ms + com0_ms
    online_time_ms = max((r.online_ms for r in results))
    shares = {party_id(r.rank): r.product_share for r in results}
    reconstructed = robust_reconstruct(shares, mpc.t, mpc.t)
    return {'n': n, 't': mpc.t, 'q': 3 * mpc.t + 1, 'rg_prep_ms': rg_prep_ms, 'com0_ms': com0_ms, 'prep_time_ms': prep_time_ms, 'online_time_ms': online_time_ms, 'expected': mod(x * y), 'reconstructed': reconstructed}

def _latency_row(n: int, t: int, q: int, repeat_id: int, *, prep_time_ms: float=0.0, online_time_ms: float=0.0, success: bool=False, error: str='') -> dict[str, Any]:
    total = float(prep_time_ms) + float(online_time_ms)
    return {'protocol': PROTOCOL, 'n': n, 't': t, 'q': q, 'repeat_id': repeat_id, 'prep_time_ms': round(float(prep_time_ms), 3), 'online_time_ms': round(float(online_time_ms), 3), 'total_time_ms': round(total, 3), **_logical_columns(_mul_prep_logical_cost(n, t), _mul_online_logical_cost(n, t)), 'success': success, 'error': error}

def run_single(n: int, *, repeat_id: int, base_port: int, startup_delay: float, timeout: float, recv_timeout: float, warmup_round: bool=True, hosts: tuple[str, ...] | None=None, remote_repo: str | None=None, ssh_user: str | None=None, shared_tmp_dir: Path | None=None) -> dict[str, Any]:
    t = validate_party_count(n)
    q = 3 * t + 1
    x, y = (rand_field(), rand_field())
    report = run_mul_standalone_multiprocess(n=n, t=t, base_port=base_port, x=x, y=y, instance_id=f'mul-standalone-n{n}-rep{repeat_id}', startup_delay=startup_delay, timeout=timeout, recv_timeout=recv_timeout, warmup_round=warmup_round, hosts=hosts, remote_repo=remote_repo, ssh_user=ssh_user, shared_tmp_dir=shared_tmp_dir)
    success = report['reconstructed'] == report['expected']
    return _latency_row(n, t, q, repeat_id, prep_time_ms=float(report['prep_time_ms']), online_time_ms=float(report['online_time_ms']), success=success)

def run_benchmark(n_values: list[int], *, repeat: int, base_port: int, startup_delay: float, timeout: float, recv_timeout: float, cooldown: float, warmup: bool=True, warmup_pause: float=1.0, warmup_round: bool=True, hosts: tuple[str, ...] | None=None, remote_repo: str | None=None, ssh_user: str | None=None, shared_tmp_dir: Path | None=None) -> list[dict[str, Any]]:
    from cubempc.benchmarks.benchmark_warmup import run_external_warmup
    rows: list[dict[str, Any]] = []
    port_cursor = 0
    used_bases: set[int] = set()
    for n in n_values:
        t = default_threshold(n)
        q = 3 * t + 1
        try:
            t = validate_party_count(n)
            q = 3 * t + 1
        except ValueError as exc:
            for repeat_id in range(repeat):
                rows.append(_latency_row(n, t, q, repeat_id, error=str(exc)))
            continue
        if warmup:
            warmup_port = _next_free_base_port(base_port + port_cursor, n, used_bases)
            used_bases.add(warmup_port)
            port_cursor += PORT_STRIDE
            run_external_warmup(run_single, warmup=True, pause=warmup_pause, n=n, repeat_id=-1, base_port=warmup_port, startup_delay=startup_delay, timeout=timeout, recv_timeout=recv_timeout, warmup_round=warmup_round, hosts=hosts, remote_repo=remote_repo, ssh_user=ssh_user, shared_tmp_dir=shared_tmp_dir)
        for repeat_id in range(repeat):
            attempts = 0
            while True:
                run_base_port = _next_free_base_port(base_port + port_cursor, n, used_bases)
                used_bases.add(run_base_port)
                port_cursor += PORT_STRIDE
                try:
                    row = run_single(n, repeat_id=repeat_id, base_port=run_base_port, startup_delay=startup_delay, timeout=timeout, recv_timeout=recv_timeout, warmup_round=warmup_round, hosts=hosts, remote_repo=remote_repo, ssh_user=ssh_user, shared_tmp_dir=shared_tmp_dir)
                    break
                except Exception as exc:
                    attempts += 1
                    if 'address already in use' not in str(exc).lower() or attempts >= 3:
                        row = _latency_row(n, t, q, repeat_id, error=str(exc))
                        break
            rows.append(row)
            if cooldown > 0:
                time.sleep(cooldown)
    return rows

def write_raw_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    from cubempc.csv_io import write_aligned_csv
    write_aligned_csv(path, RAW_FIELDS, rows)

def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 3) if values else 0.0

def _std(values: list[float]) -> float:
    return round(statistics.stdev(values), 3) if len(values) > 1 else 0.0

def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row['protocol'], int(row['n']), int(row['t']), int(row['q']))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (protocol, n, t, q), group in sorted(groups.items()):
        successes = [row for row in group if bool(row['success'])]
        prep = [float(row['prep_time_ms']) for row in successes]
        online = [float(row['online_time_ms']) for row in successes]
        total = [float(row['total_time_ms']) for row in successes]
        logical_values = {field: [float(row[field]) for row in successes] for field in LOGICAL_RAW_FIELDS}
        success_count = len(successes)
        total_count = len(group)
        summary_row = {'protocol': protocol, 'n': n, 't': t, 'q': q, 'success_count': success_count, 'total_count': total_count, 'success_rate': round(success_count / total_count, 6) if total_count else 0.0, 'prep_time_ms_mean': _mean(prep), 'prep_time_ms_std': _std(prep), 'online_time_ms_mean': _mean(online), 'online_time_ms_std': _std(online), 'total_time_ms_mean': _mean(total), 'total_time_ms_std': _std(total)}
        summary_row.update({f'{field}_mean': _mean(values) for field, values in logical_values.items()})
        summary.append(summary_row)
    return summary

def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    from cubempc.csv_io import write_aligned_csv
    write_aligned_csv(path, SUMMARY_FIELDS, rows)

def main(argv: list[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description='Latency-only standalone ΠMUL benchmark')
    parser.add_argument('--n-list', type=int, nargs='+', default=[5])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=50000)
    parser.add_argument('--startup-delay', type=float, default=2.5)
    parser.add_argument('--timeout', type=float, default=7200.0)
    parser.add_argument('--recv-timeout', type=float, default=120.0)
    parser.add_argument('--cooldown', type=float, default=1.0)
    parser.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--warmup-pause', type=float, default=1.0)
    parser.add_argument('--warmup-round', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--raw-out', type=Path, default=Path('bench_output/mul_standalone_raw.csv'))
    parser.add_argument('--hosts', type=str, default=None, help='comma-separated hostnames/IPs for multi-server rank placement')
    parser.add_argument('--remote-repo-dir', type=str, default=None, help='path to repo on each remote host when --hosts is distributed')
    parser.add_argument('--ssh-user', type=str, default=None)
    parser.add_argument('--shared-tmp-dir', type=Path, default=None, help='shared directory visible at the same path on all hosts')
    args = parser.parse_args(argv)
    host_layout = HostLayout.parse(args.hosts)
    rows = run_benchmark(args.n_list, repeat=args.repeat, base_port=args.base_port, startup_delay=args.startup_delay, timeout=args.timeout, recv_timeout=args.recv_timeout, cooldown=args.cooldown, warmup=args.warmup, warmup_pause=args.warmup_pause, warmup_round=args.warmup_round, hosts=host_layout.hosts, remote_repo=args.remote_repo_dir, ssh_user=args.ssh_user, shared_tmp_dir=args.shared_tmp_dir)
    write_raw_csv(args.raw_out, rows)
    print(f'wrote {args.raw_out} ({len(rows)} rows)')
if __name__ == '__main__':
    main()