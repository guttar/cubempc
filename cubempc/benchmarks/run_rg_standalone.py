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
from cubempc.host_layout import HostLayout
from cubempc.metrics import rg_logical_cost
from cubempc.network.coordinator import join_worker_processes
from cubempc.network.node import NodeProcess
from cubempc.protocols.rg import run_rg, run_rg_batch
from cubempc.protocols.vss import party_id, run_discard_vss_warmup
from cubempc.shamir import robust_reconstruct
from cubempc.vss_profiling import VssProfiler, set_profiler
PROTOCOL = 'RG'
RG_BATCH_SUMMARY_FIELDS = ['protocol', 'n', 't', 'batch_size', 'repeat_id', 'wall_ms', 'success', 'error']
VSS_STAGE_PROFILE_FIELDS = ['scheme', 'n', 't', 'd', 'w', 'repeat_id', 'phase', 'layer_id', 'stage', 'rank_id', 'op_name', 'op_count', 'cpu_ms', 'wall_ms']
LOGICAL_RAW_FIELDS = ['prep_logical_p2p_bytes', 'prep_logical_broadcast_bytes', 'prep_logical_total_bytes', 'prep_logical_message_count', 'online_logical_p2p_bytes', 'online_logical_broadcast_bytes', 'online_logical_total_bytes', 'online_logical_message_count', 'logical_p2p_bytes', 'logical_broadcast_bytes', 'logical_total_bytes', 'logical_message_count']
RAW_FIELDS = ['protocol', 'n', 't', 'q', 'repeat_id', 'prep_time_ms', 'online_time_ms', 'total_time_ms', *LOGICAL_RAW_FIELDS, 'success', 'error']
SUMMARY_FIELDS = ['protocol', 'n', 't', 'q', 'success_count', 'total_count', 'success_rate', 'prep_time_ms_mean', 'prep_time_ms_std', 'online_time_ms_mean', 'online_time_ms_std', 'total_time_ms_mean', 'total_time_ms_std', *(f'{field}_mean' for field in LOGICAL_RAW_FIELDS)]
PORT_STRIDE = 1000

def _zero_logical_cost() -> dict[str, int]:
    return {'logical_p2p_bytes': 0, 'logical_broadcast_bytes': 0, 'logical_total_bytes': 0, 'logical_message_count': 0}

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
class RGStandaloneNodeResult:
    rank: int
    pid: int
    shares: list[int]
    protocol_time_ms: float
    vss_samples: list[dict[str, Any]] | None = None

async def _run_rg_standalone_node_async(rank: int, n: int, t: int | None, base_port: int, start_layer: int, q: int, instance_id: str, startup_delay: float, warmup_round: bool, *, batch_size: int | None=None, enable_vss_profiling: bool=False, hosts: tuple[str, ...] | None=None) -> RGStandaloneNodeResult:
    import os
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    node = NodeProcess(rank, n, base_port, mpc)
    await node.start_server()
    await asyncio.sleep(startup_delay)
    await node.connect_peers()
    seed = int.from_bytes(hashlib.sha256(f'{instance_id}:{rank}'.encode()).digest()[:8], 'big')
    rng = Random(seed)
    loop = asyncio.get_running_loop()
    if warmup_round:
        await run_discard_vss_warmup(node, start_layer, instance_id, rng)
    vss_profiler = VssProfiler(enabled=enable_vss_profiling and batch_size is not None, rank_id=rank, phase='LAYER_RG_PREP', layer_id=0, stage='')
    set_profiler(vss_profiler)
    t0 = loop.time()
    vss_samples: list[dict[str, Any]] | None = None
    if batch_size is not None:
        records = await run_rg_batch(node, start_layer, batch_size, instance_id, rng=rng, warmup_round=False)
        shares = [record.values[0] if record.values else 0 for record in records]
        if enable_vss_profiling:
            vss_samples = [{'scheme': 'CUBE', 'n': n, 't': mpc.t, 'd': 1, 'w': batch_size, 'repeat_id': -1, 'phase': sample.phase, 'layer_id': sample.layer_id, 'stage': sample.stage, 'rank_id': sample.rank_id, 'op_name': sample.op_name, 'op_count': sample.op_count, 'cpu_ms': round(sample.cpu_ms, 6), 'wall_ms': round(sample.wall_ms, 6)} for sample in vss_profiler.samples]
    else:
        shares = await run_rg(node, start_layer, q, instance_id, rng=rng, warmup_round=False)
    protocol_time_ms = (loop.time() - t0) * 1000.0
    await node.stop_server()
    return RGStandaloneNodeResult(rank=rank, pid=os.getpid(), shares=shares, protocol_time_ms=protocol_time_ms, vss_samples=vss_samples)

def _rg_standalone_worker_entry(*args: object) -> None:
    import traceback
    from queue import Queue
    rank, n, t, base_port, start_layer, q, instance_id, startup_delay, warmup_round, batch_size, enable_vss_profiling, hosts, out_queue = args
    q_out: Queue = out_queue
    try:
        result = asyncio.run(_run_rg_standalone_node_async(int(rank), int(n), t, int(base_port), int(start_layer), int(q), str(instance_id), float(startup_delay), bool(warmup_round), batch_size=int(batch_size) if batch_size is not None else None, enable_vss_profiling=bool(enable_vss_profiling), hosts=hosts if hosts is None else tuple(hosts)))
        q_out.put(result)
    except Exception:
        q_out.put({'rank': rank, 'error': traceback.format_exc()})

def _reconstruct_rg_shares(per_rank_shares: dict[int, list[int]], q: int, t: int) -> list[int | None]:
    reconstructed: list[int | None] = []
    for ell in range(q):
        shares_ell = {party: per_rank_shares[party][ell] for party in per_rank_shares}
        reconstructed.append(robust_reconstruct(shares_ell, t, t))
    return reconstructed

def run_rg_standalone_multiprocess(n: int, *, t: int | None, base_port: int, start_layer: int=0, instance_id: str, startup_delay: float, timeout: float, warmup_round: bool=False, batch_size: int | None=None, enable_vss_profiling: bool=False, hosts: tuple[str, ...] | None=None, remote_repo: str | None=None, ssh_user: str | None=None, shared_tmp_dir: Path | None=None) -> dict[str, Any]:
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    q = n - mpc.t
    results: list[RGStandaloneNodeResult] = []
    first_error: str | None = None
    layout = mpc.host_layout()
    if distributed_layout(layout):
        if not remote_repo:
            raise ValueError('--remote-repo-dir is required when --hosts spans multiple machines')
        if shared_tmp_dir is None:
            raise ValueError('--shared-tmp-dir is required when --hosts spans multiple machines')
        with tempfile.TemporaryDirectory(prefix='rg-standalone-') as tmp:
            _ = tmp
            work_dir = shared_tmp_dir / instance_id
            work_dir.mkdir(parents=True, exist_ok=True)
            procs: list[Any] = []
            result_paths: list[Path] = []
            for rank in range(n):
                params_path = work_dir / f'rank{rank}-params.pkl'
                result_path = work_dir / f'rank{rank}-result.pkl'
                params = {'module': __name__, 'function': '_run_rg_standalone_node_async', 'kwargs': {'rank': rank, 'n': n, 't': t, 'base_port': base_port, 'start_layer': start_layer, 'q': q, 'instance_id': instance_id, 'startup_delay': startup_delay, 'warmup_round': warmup_round, 'batch_size': batch_size, 'enable_vss_profiling': enable_vss_profiling, 'hosts': hosts}}
                with params_path.open('wb') as fh:
                    pickle.dump(params, fh)
                procs.append(launch_remote_rank(rank=rank, host=layout.host_for_rank(rank), remote_repo=remote_repo, python_executable=sys.executable, worker_args=['--params-file', str(params_path), '--result-file', str(result_path)], ssh_user=ssh_user, worker_module='cubempc.benchmarks.subprotocol_worker'))
                result_paths.append(result_path)
            wait_for_result_files(result_paths, timeout=timeout)
            for proc in procs:
                rc = proc.wait(timeout=60)
                if rc != 0 and first_error is None:
                    first_error = f'remote RG worker exit code {rc}'
            for result_path in result_paths:
                item = load_worker_result(result_path)
                if isinstance(item, RGStandaloneNodeResult):
                    results.append(item)
                elif isinstance(item, dict) and 'error' in item:
                    first_error = f'RG node {item.get('rank')} failed: {item['error']}'
                    break
    else:
        ctx = mp.get_context('spawn')
        out_queue: mp.Queue = ctx.Queue()
        processes: list[mp.Process] = []
        for rank in range(n):
            proc = ctx.Process(target=_rg_standalone_worker_entry, args=(rank, n, t, base_port, start_layer, q, instance_id, startup_delay, warmup_round, batch_size, enable_vss_profiling, hosts, out_queue), name=f'rg-standalone-{rank}')
            proc.start()
            processes.append(proc)
        deadline = time.time() + timeout
        while len(results) < n and time.time() < deadline:
            try:
                item = out_queue.get(timeout=0.5)
                if isinstance(item, RGStandaloneNodeResult):
                    results.append(item)
                elif isinstance(item, dict) and 'error' in item:
                    first_error = f'RG node {item.get('rank')} failed: {item['error']}'
                    break
            except Empty:
                continue
        join_worker_processes(processes)
    if first_error is not None:
        raise RuntimeError(first_error)
    if len(results) < n:
        raise RuntimeError(f'expected {n} RG results, got {len(results)}')
    results.sort(key=lambda r: r.rank)
    per_rank_shares = {party_id(r.rank): r.shares for r in results}
    if batch_size is not None:
        reconstructed = None
        success = all((len(r.shares) == batch_size for r in results))
    else:
        reconstructed = _reconstruct_rg_shares(per_rank_shares, q, mpc.t)
        success = len(reconstructed) == q and all((value is not None for value in reconstructed))
    total_time_ms = max((r.protocol_time_ms for r in results))
    vss_profile_rows: list[dict[str, Any]] = []
    if enable_vss_profiling:
        for result in results:
            if result.vss_samples:
                vss_profile_rows.extend(result.vss_samples)
    return {'n': n, 't': mpc.t, 'q': q, 'batch_size': batch_size, 'success': success, 'reconstructed': reconstructed, 'per_rank_shares': per_rank_shares, 'per_rank': [r.__dict__ for r in results], 'total_time_ms': total_time_ms, 'vss_profile_rows': vss_profile_rows}
run_rg_parallel_equiv_multiprocess = run_rg_standalone_multiprocess

def _latency_row(n: int, t: int, q: int, repeat_id: int, *, online_time_ms: float=0.0, success: bool=False, error: str='') -> dict[str, Any]:
    return {'protocol': PROTOCOL, 'n': n, 't': t, 'q': q, 'repeat_id': repeat_id, 'prep_time_ms': 0.0, 'online_time_ms': round(float(online_time_ms), 3), 'total_time_ms': round(float(online_time_ms), 3), **_logical_columns(_zero_logical_cost(), rg_logical_cost(n, t)), 'success': success, 'error': error}

def run_single(n: int, *, repeat_id: int, base_port: int, startup_delay: float, timeout: float, warmup_round: bool=True, batch_size: int | None=None, enable_vss_profiling: bool=False, hosts: tuple[str, ...] | None=None, remote_repo: str | None=None, ssh_user: str | None=None, shared_tmp_dir: Path | None=None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t = validate_party_count(n)
    q = n - t
    report = run_rg_standalone_multiprocess(n=n, t=t, base_port=base_port, instance_id=f'rg-standalone-n{n}-rep{repeat_id}', startup_delay=startup_delay, timeout=timeout, warmup_round=warmup_round, batch_size=batch_size, enable_vss_profiling=enable_vss_profiling, hosts=hosts, remote_repo=remote_repo, ssh_user=ssh_user, shared_tmp_dir=shared_tmp_dir)
    success = bool(report['success'])
    row = _latency_row(n, t, q, repeat_id, online_time_ms=float(report['total_time_ms']), success=success)
    profile_rows = list(report.get('vss_profile_rows') or [])
    for profile_row in profile_rows:
        profile_row['repeat_id'] = repeat_id
    return (row, profile_rows)

def _rg_batch_summary_row(n: int, t: int, batch_size: int, repeat_id: int, *, wall_ms: float, success: bool, error: str='') -> dict[str, Any]:
    return {'protocol': PROTOCOL, 'n': n, 't': t, 'batch_size': batch_size, 'repeat_id': repeat_id, 'wall_ms': round(wall_ms, 3), 'success': success, 'error': error}

def run_benchmark(n_values: list[int], *, repeat: int, base_port: int, startup_delay: float, timeout: float, cooldown: float, warmup: bool=True, warmup_pause: float=1.0, warmup_round: bool=True, batch_size: int | None=None, enable_vss_profiling: bool=False, hosts: tuple[str, ...] | None=None, remote_repo: str | None=None, ssh_user: str | None=None, shared_tmp_dir: Path | None=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from cubempc.benchmarks.benchmark_warmup import run_external_warmup
    rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    port_cursor = 0
    used_bases: set[int] = set()
    for n in n_values:
        t = default_threshold(n)
        q = max(0, n - t)
        try:
            t = validate_party_count(n)
            q = n - t
        except ValueError as exc:
            for repeat_id in range(repeat):
                rows.append(_latency_row(n, t, q, repeat_id, error=str(exc)))
            continue
        if warmup:
            warmup_port = _next_free_base_port(base_port + port_cursor, n, used_bases)
            used_bases.add(warmup_port)
            port_cursor += PORT_STRIDE
            run_external_warmup(run_single, warmup=True, pause=warmup_pause, n=n, repeat_id=-1, base_port=warmup_port, startup_delay=startup_delay, timeout=timeout, warmup_round=warmup_round, hosts=hosts, remote_repo=remote_repo, ssh_user=ssh_user, shared_tmp_dir=shared_tmp_dir)
        for repeat_id in range(repeat):
            run_base_port = _next_free_base_port(base_port + port_cursor, n, used_bases)
            used_bases.add(run_base_port)
            port_cursor += PORT_STRIDE
            try:
                row, run_profile_rows = run_single(n, repeat_id=repeat_id, base_port=run_base_port, startup_delay=startup_delay, timeout=timeout, warmup_round=warmup_round, batch_size=batch_size, enable_vss_profiling=enable_vss_profiling, hosts=hosts, remote_repo=remote_repo, ssh_user=ssh_user, shared_tmp_dir=shared_tmp_dir)
                profile_rows.extend(run_profile_rows)
                if batch_size is not None:
                    batch_rows.append(_rg_batch_summary_row(n, t, batch_size, repeat_id, wall_ms=float(row['total_time_ms']), success=bool(row['success'])))
            except Exception as exc:
                row = _latency_row(n, t, q, repeat_id, error=str(exc))
                if batch_size is not None:
                    batch_rows.append(_rg_batch_summary_row(n, t, batch_size, repeat_id, wall_ms=0.0, success=False, error=str(exc)))
            rows.append(row)
            if cooldown > 0:
                time.sleep(cooldown)
    return (rows, batch_rows, profile_rows)

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

def _parse_n_values(raw: str | None, n_list: list[int] | None) -> list[int]:
    if raw is not None:
        return [int(part.strip()) for part in raw.split(',') if part.strip()]
    return n_list or [5]

def main(argv: list[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description='Latency-only standalone ΠRG benchmark')
    parser.add_argument('--n-list', type=int, nargs='+', default=None)
    parser.add_argument('--n', type=str, default=None, help='comma-separated party counts, e.g. 5,9')
    parser.add_argument('--batch-size', type=int, default=None, help='batched RG size (enables run_rg_batch micro benchmark)')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=30000)
    parser.add_argument('--startup-delay', type=float, default=2.5)
    parser.add_argument('--timeout', type=float, default=7200.0)
    parser.add_argument('--cooldown', type=float, default=1.0)
    parser.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True, help='run one untimed invocation per n before timed repeats')
    parser.add_argument('--warmup-pause', type=float, default=1.0)
    parser.add_argument('--warmup-round', action=argparse.BooleanOptionalAction, default=True, help='discard one in-process VSS round before measured RG prep')
    parser.add_argument('--raw-out', type=Path, default=Path('bench_output/rg_standalone_raw.csv'))
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--hosts', type=str, default=None, help='comma-separated hostnames/IPs for multi-server rank placement')
    parser.add_argument('--remote-repo-dir', type=str, default=None, help='path to repo on each remote host when --hosts is distributed')
    parser.add_argument('--ssh-user', type=str, default=None)
    parser.add_argument('--shared-tmp-dir', type=Path, default=None, help='shared directory visible at the same path on all hosts')
    args = parser.parse_args(argv)
    n_values = _parse_n_values(args.n, args.n_list)
    host_layout = HostLayout.parse(args.hosts)
    out_dir = args.out_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_out = out_dir / 'rg_standalone_raw.csv'
        batch_out = out_dir / 'rg_batch_summary.csv'
        profile_out = out_dir / 'vss_stage_profile.csv'
    else:
        raw_out = args.raw_out
        batch_out = None
        profile_out = None
    rows, batch_rows, profile_rows = run_benchmark(n_values, repeat=args.repeat, base_port=args.base_port, startup_delay=args.startup_delay, timeout=args.timeout, cooldown=args.cooldown, warmup=args.warmup, warmup_pause=args.warmup_pause, warmup_round=args.warmup_round, batch_size=args.batch_size, enable_vss_profiling=args.batch_size is not None, hosts=host_layout.hosts, remote_repo=args.remote_repo_dir, ssh_user=args.ssh_user, shared_tmp_dir=args.shared_tmp_dir)
    write_raw_csv(raw_out, rows)
    print(f'wrote {raw_out} ({len(rows)} rows)')
    if batch_out is not None and batch_rows:
        from cubempc.csv_io import write_aligned_csv
        write_aligned_csv(batch_out, RG_BATCH_SUMMARY_FIELDS, batch_rows)
        print(f'wrote {batch_out} ({len(batch_rows)} rows)')
    if profile_out is not None and profile_rows:
        from cubempc.csv_io import write_aligned_csv
        write_aligned_csv(profile_out, VSS_STAGE_PROFILE_FIELDS, profile_rows)
        print(f'wrote {profile_out} ({len(profile_rows)} rows)')
if __name__ == '__main__':
    main()