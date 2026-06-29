from __future__ import annotations
import argparse
import csv
import socket
import statistics
import time
from pathlib import Path
from typing import Any
from cubempc.benchmarks.benchmark_warmup import run_external_warmup
from cubempc.config import default_threshold, validate_party_count
from cubempc.field import mod, rand_field
from cubempc.metrics import vss_logical_cost
from cubempc.protocols.vss import run_vss_multiprocess
PROTOCOL = 'VSS'
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

def _latency_row(n: int, t: int, repeat_id: int, *, online_time_ms: float=0.0, success: bool=False, error: str='') -> dict[str, Any]:
    return {'protocol': PROTOCOL, 'n': n, 't': t, 'q': 0, 'repeat_id': repeat_id, 'prep_time_ms': 0.0, 'online_time_ms': round(float(online_time_ms), 3), 'total_time_ms': round(float(online_time_ms), 3), **_logical_columns(_zero_logical_cost(), vss_logical_cost(n, t)), 'success': success, 'error': error}

def run_single(n: int, *, repeat_id: int, base_port: int, startup_delay: float, timeout: float, warmup_round: bool=True) -> dict[str, Any]:
    t = validate_party_count(n)
    secret = rand_field()
    report = run_vss_multiprocess(n=n, secret=secret, t=t, base_port=base_port, instance_id=f'vss-standalone-n{n}-rep{repeat_id}', startup_delay=startup_delay, timeout=timeout, warmup_round=warmup_round)
    online_time_ms = float(report['total_time_ms'])
    success = report['reconstructed'] == mod(secret)
    return _latency_row(n, t, repeat_id, online_time_ms=online_time_ms, success=success)

def run_benchmark(n_values: list[int], *, repeat: int, base_port: int, startup_delay: float, timeout: float, cooldown: float, warmup: bool=True, warmup_pause: float=1.0, warmup_round: bool=True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    port_cursor = 0
    used_bases: set[int] = set()
    for n in n_values:
        t = default_threshold(n)
        try:
            t = validate_party_count(n)
        except ValueError as exc:
            for repeat_id in range(repeat):
                rows.append(_latency_row(n, t, repeat_id, error=str(exc)))
            continue
        if warmup:
            warmup_port = _next_free_base_port(base_port + port_cursor, n, used_bases)
            used_bases.add(warmup_port)
            port_cursor += PORT_STRIDE
            run_external_warmup(run_single, warmup=True, pause=warmup_pause, n=n, repeat_id=-1, base_port=warmup_port, startup_delay=startup_delay, timeout=timeout, warmup_round=warmup_round)
        for repeat_id in range(repeat):
            run_base_port = _next_free_base_port(base_port + port_cursor, n, used_bases)
            used_bases.add(run_base_port)
            port_cursor += PORT_STRIDE
            try:
                row = run_single(n, repeat_id=repeat_id, base_port=run_base_port, startup_delay=startup_delay, timeout=timeout, warmup_round=warmup_round)
            except Exception as exc:
                row = _latency_row(n, t, repeat_id, error=str(exc))
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
    parser = argparse.ArgumentParser(description='Latency-only standalone ΠVSS benchmark')
    parser.add_argument('--n-list', type=int, nargs='+', default=[5])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=20000)
    parser.add_argument('--startup-delay', type=float, default=2.5)
    parser.add_argument('--timeout', type=float, default=900.0)
    parser.add_argument('--cooldown', type=float, default=1.0)
    parser.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--warmup-pause', type=float, default=1.0)
    parser.add_argument('--warmup-round', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--raw-out', type=Path, default=Path('bench_output/vss_standalone_raw.csv'))
    args = parser.parse_args(argv)
    rows = run_benchmark(args.n_list, repeat=args.repeat, base_port=args.base_port, startup_delay=args.startup_delay, timeout=args.timeout, cooldown=args.cooldown, warmup=args.warmup, warmup_pause=args.warmup_pause, warmup_round=args.warmup_round)
    write_raw_csv(args.raw_out, rows)
    print(f'wrote {args.raw_out} ({len(rows)} rows)')
if __name__ == '__main__':
    main()