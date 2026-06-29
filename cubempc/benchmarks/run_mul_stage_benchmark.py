from __future__ import annotations
import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any
from cubempc.benchmarks.run_mul_standalone import PORT_STRIDE, _next_free_base_port, run_mul_standalone_multiprocess
from cubempc.config import default_threshold, validate_party_count
from cubempc.field import rand_field
PROTOCOL = 'MUL'
RAW_FIELDS = ['protocol', 'n', 't', 'q', 'repeat_id', 'rg_prep_ms', 'com0_ms', 'prep_time_ms', 'online_time_ms', 'total_time_ms', 'success', 'error']
SUMMARY_FIELDS = ['protocol', 'n', 't', 'q', 'success_count', 'total_count', 'success_rate', 'rg_prep_ms_mean', 'rg_prep_ms_std', 'com0_ms_mean', 'com0_ms_std', 'prep_time_ms_mean', 'prep_time_ms_std', 'online_time_ms_mean', 'online_time_ms_std', 'total_time_ms_mean', 'total_time_ms_std']

def _stage_row(n: int, t: int, q: int, repeat_id: int, *, rg_prep_ms: float=0.0, com0_ms: float=0.0, prep_time_ms: float=0.0, online_time_ms: float=0.0, success: bool=False, error: str='') -> dict[str, Any]:
    total = float(prep_time_ms) + float(online_time_ms)
    return {'protocol': PROTOCOL, 'n': n, 't': t, 'q': q, 'repeat_id': repeat_id, 'rg_prep_ms': round(float(rg_prep_ms), 3), 'com0_ms': round(float(com0_ms), 3), 'prep_time_ms': round(float(prep_time_ms), 3), 'online_time_ms': round(float(online_time_ms), 3), 'total_time_ms': round(total, 3), 'success': success, 'error': error}

def run_single(n: int, *, repeat_id: int, base_port: int, startup_delay: float, timeout: float, recv_timeout: float, warmup_round: bool=True) -> dict[str, Any]:
    t = validate_party_count(n)
    q = 3 * t + 1
    x, y = (rand_field(), rand_field())
    report = run_mul_standalone_multiprocess(n=n, t=t, base_port=base_port, x=x, y=y, instance_id=f'mul-stage-n{n}-rep{repeat_id}', startup_delay=startup_delay, timeout=timeout, recv_timeout=recv_timeout, warmup_round=warmup_round)
    success = report['reconstructed'] == report['expected']
    return _stage_row(n, t, q, repeat_id, rg_prep_ms=float(report['rg_prep_ms']), com0_ms=float(report['com0_ms']), prep_time_ms=float(report['prep_time_ms']), online_time_ms=float(report['online_time_ms']), success=success)

def run_benchmark(n_values: list[int], *, repeat: int, base_port: int, startup_delay: float, timeout: float, recv_timeout: float, cooldown: float, warmup: bool=True, warmup_pause: float=1.0, warmup_round: bool=True) -> list[dict[str, Any]]:
    from cubempc.benchmarks.benchmark_warmup import run_external_warmup
    rows: list[dict[str, Any]] = []
    port_cursor = 0
    used_bases: set[int] = set()
    for n in n_values:
        try:
            t = validate_party_count(n)
            q = 3 * t + 1
        except ValueError as exc:
            for repeat_id in range(repeat):
                rows.append(_stage_row(n, default_threshold(n), 0, repeat_id, error=str(exc)))
            continue
        if warmup:
            warmup_port = _next_free_base_port(base_port + port_cursor, n, used_bases)
            used_bases.add(warmup_port)
            port_cursor += PORT_STRIDE
            run_external_warmup(run_single, warmup=True, pause=warmup_pause, n=n, repeat_id=-1, base_port=warmup_port, startup_delay=startup_delay, timeout=timeout, recv_timeout=recv_timeout, warmup_round=warmup_round)
        for repeat_id in range(repeat):
            attempts = 0
            while True:
                run_base_port = _next_free_base_port(base_port + port_cursor, n, used_bases)
                used_bases.add(run_base_port)
                port_cursor += PORT_STRIDE
                try:
                    row = run_single(n, repeat_id=repeat_id, base_port=run_base_port, startup_delay=startup_delay, timeout=timeout, recv_timeout=recv_timeout, warmup_round=warmup_round)
                    break
                except Exception as exc:
                    attempts += 1
                    if 'address already in use' not in str(exc).lower() or attempts >= 3:
                        row = _stage_row(n, t, q, repeat_id, error=str(exc))
                        break
            rows.append(row)
            if cooldown > 0:
                time.sleep(cooldown)
    return rows

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    from cubempc.csv_io import write_aligned_csv
    write_aligned_csv(path, fields, rows)

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
        rg = [float(row['rg_prep_ms']) for row in successes]
        com0 = [float(row['com0_ms']) for row in successes]
        prep = [float(row['prep_time_ms']) for row in successes]
        online = [float(row['online_time_ms']) for row in successes]
        total = [float(row['total_time_ms']) for row in successes]
        success_count = len(successes)
        total_count = len(group)
        summary.append({'protocol': protocol, 'n': n, 't': t, 'q': q, 'success_count': success_count, 'total_count': total_count, 'success_rate': round(success_count / total_count, 6) if total_count else 0.0, 'rg_prep_ms_mean': _mean(rg), 'rg_prep_ms_std': _std(rg), 'com0_ms_mean': _mean(com0), 'com0_ms_std': _std(com0), 'prep_time_ms_mean': _mean(prep), 'prep_time_ms_std': _std(prep), 'online_time_ms_mean': _mean(online), 'online_time_ms_std': _std(online), 'total_time_ms_mean': _mean(total), 'total_time_ms_std': _std(total)})
    return summary

def main(argv: list[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description='Per-stage ΠMUL latency benchmark')
    parser.add_argument('--n-list', type=int, nargs='+', default=[5])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=51000)
    parser.add_argument('--startup-delay', type=float, default=2.5)
    parser.add_argument('--timeout', type=float, default=7200.0)
    parser.add_argument('--recv-timeout', type=float, default=120.0)
    parser.add_argument('--cooldown', type=float, default=1.0)
    parser.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--warmup-pause', type=float, default=1.0)
    parser.add_argument('--warmup-round', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--out', type=Path, default=Path('bench_output/mul_stage_by_n.csv'))
    args = parser.parse_args(argv)
    rows = run_benchmark(args.n_list, repeat=args.repeat, base_port=args.base_port, startup_delay=args.startup_delay, timeout=args.timeout, recv_timeout=args.recv_timeout, cooldown=args.cooldown, warmup=args.warmup, warmup_pause=args.warmup_pause, warmup_round=args.warmup_round)
    write_csv(args.out, rows, RAW_FIELDS)
    print(f'wrote {args.out} ({len(rows)} rows)')
if __name__ == '__main__':
    main()