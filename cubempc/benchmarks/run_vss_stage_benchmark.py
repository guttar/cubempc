from __future__ import annotations
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any
from cubempc.config import default_threshold, validate_party_count
from cubempc.field import mod, rand_field
from cubempc.metrics import vss_stage_logical_cost
from cubempc.protocols.vss import VSSNodeResult, aggregate_stage_metrics, run_vss_multiprocess
PROTOCOL_NAME = 'VSS'
STAGE_ORDER = ('Dealer', 'Com1', 'Com2Decode', 'Com2Broadcast', 'Com2SendScalar', 'Public', 'Com3', 'Total')
STAGE_LATENCY_KEYS: dict[str, str] = {'Dealer': 'dealer_ms', 'Com1': 'com1_ms', 'Com2Decode': 'com2_decode_ms', 'Com2Broadcast': 'com2_broadcast_ms', 'Com2SendScalar': 'com2_send_scalar_ms', 'Public': 'public_ms', 'Com3': 'com3_ms', 'Total': 'total_time_ms'}
LONG_CSV_FIELDS = ['protocol', 'n', 't', 'repeat_id', 'stage', 'latency_ms', 'p2p_bytes', 'broadcast_bytes', 'total_bytes', 'message_count', 'success']
WIDE_CSV_FIELDS = ['protocol', 'n', 't', 'repeat_id', 'dealer_ms', 'dealer_p2p_kb', 'com1_ms', 'com1_p2p_kb', 'com2_decode_ms', 'com2_broadcast_ms', 'com2_broadcast_kb', 'com2_send_scalar_ms', 'com2_send_scalar_p2p_kb', 'public_ms', 'com3_ms', 'total_time_ms', 'total_p2p_kb', 'total_broadcast_kb', 'total_kb', 'message_count', 'success']
LOG_PATH = Path('logs/vss_stage_benchmark.log')

def _setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('vss_stage_benchmark')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def _parse_n_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError('empty --n list')
    return out

def _bytes_to_kb(n_bytes: int) -> float:
    return round(n_bytes / 1024.0, 6)

def _aggregate_from_report(report: dict[str, Any], *, dealer_rank: int=0) -> tuple[dict[str, float], list[VSSNodeResult]]:
    per_rank = report.get('per_rank', [])
    results = [VSSNodeResult(**row) for row in per_rank]
    stage_latencies = aggregate_stage_metrics(results, dealer_rank=dealer_rank)
    stage_latencies['total_time_ms'] = float(report.get('total_time_ms', 0.0))
    return (stage_latencies, results)

def run_single_vss_stage_benchmark(n: int, *, repeat_id: int, base_port: int, dealer_rank: int=0, startup_delay: float=2.0, timeout: float=300.0, start_layer: int=0) -> dict[str, Any]:
    t = validate_party_count(n)
    secret = rand_field()
    instance_id = f'vss-stage-n{n}-rep{repeat_id}'
    report = run_vss_multiprocess(n=n, secret=secret, t=t, base_port=base_port, start_layer=start_layer, dealer_rank=dealer_rank, instance_id=instance_id, startup_delay=startup_delay, timeout=timeout)
    shares = report['shares']
    reconstructed = report['reconstructed']
    secret_mod = mod(secret)
    success = reconstructed == secret_mod
    stage_latencies, _results = _aggregate_from_report(report, dealer_rank=dealer_rank)
    stage_comm = vss_stage_logical_cost(n, t)
    return {'protocol': PROTOCOL_NAME, 'n': n, 't': t, 'repeat_id': repeat_id, 'success': success, 'stage_latencies': stage_latencies, 'stage_comm': stage_comm, '_secret': secret_mod, '_reconstructed': reconstructed, '_shares': shares}

def _failure_latencies() -> dict[str, float]:
    return {key: 0.0 for key in STAGE_LATENCY_KEYS.values()}

def _failure_comm(n: int, t: int) -> dict[str, dict[str, int]]:
    return vss_stage_logical_cost(n, t)

def build_long_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    latencies = run.get('stage_latencies') or _failure_latencies()
    comm = run.get('stage_comm') or _failure_comm(run['n'], run['t'])
    success = run['success']
    rows: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        latency_key = STAGE_LATENCY_KEYS[stage]
        stage_comm = comm[stage]
        rows.append({'protocol': run['protocol'], 'n': run['n'], 't': run['t'], 'repeat_id': run['repeat_id'], 'stage': stage, 'latency_ms': round(float(latencies.get(latency_key, 0.0)), 3), 'p2p_bytes': stage_comm['p2p_bytes'], 'broadcast_bytes': stage_comm['broadcast_bytes'], 'total_bytes': stage_comm['total_bytes'], 'message_count': stage_comm['message_count'], 'success': success})
    return rows

def build_wide_row(run: dict[str, Any]) -> dict[str, Any]:
    latencies = run.get('stage_latencies') or _failure_latencies()
    comm = run.get('stage_comm') or _failure_comm(run['n'], run['t'])
    total_comm = comm['Total']
    return {'protocol': run['protocol'], 'n': run['n'], 't': run['t'], 'repeat_id': run['repeat_id'], 'dealer_ms': round(latencies.get('dealer_ms', 0.0), 3), 'dealer_p2p_kb': _bytes_to_kb(comm['Dealer']['p2p_bytes']), 'com1_ms': round(latencies.get('com1_ms', 0.0), 3), 'com1_p2p_kb': _bytes_to_kb(comm['Com1']['p2p_bytes']), 'com2_decode_ms': round(latencies.get('com2_decode_ms', 0.0), 3), 'com2_broadcast_ms': round(latencies.get('com2_broadcast_ms', 0.0), 3), 'com2_broadcast_kb': _bytes_to_kb(comm['Com2Broadcast']['broadcast_bytes']), 'com2_send_scalar_ms': round(latencies.get('com2_send_scalar_ms', 0.0), 3), 'com2_send_scalar_p2p_kb': _bytes_to_kb(comm['Com2SendScalar']['p2p_bytes']), 'public_ms': round(latencies.get('public_ms', 0.0), 3), 'com3_ms': round(latencies.get('com3_ms', 0.0), 3), 'total_time_ms': round(latencies.get('total_time_ms', 0.0), 3), 'total_p2p_kb': _bytes_to_kb(total_comm['p2p_bytes']), 'total_broadcast_kb': _bytes_to_kb(total_comm['broadcast_bytes']), 'total_kb': _bytes_to_kb(total_comm['total_bytes']), 'message_count': total_comm['message_count'], 'success': run['success']}

def write_long_csv(path: str, rows: list[dict[str, Any]]) -> None:
    from cubempc.csv_io import write_aligned_csv
    write_aligned_csv(path, LONG_CSV_FIELDS, [{k: row[k] for k in LONG_CSV_FIELDS} for row in rows])

def write_wide_csv(path: str, rows: list[dict[str, Any]]) -> None:
    from cubempc.csv_io import write_aligned_csv
    write_aligned_csv(path, WIDE_CSV_FIELDS, [{k: row[k] for k in WIDE_CSV_FIELDS} for row in rows])

def _failure_run(n: int, repeat_id: int, *, t: int | None=None) -> dict[str, Any]:
    threshold = t if t is not None else default_threshold(n)
    return {'protocol': PROTOCOL_NAME, 'n': n, 't': threshold, 'repeat_id': repeat_id, 'success': False, 'stage_latencies': _failure_latencies(), 'stage_comm': vss_stage_logical_cost(n, threshold)}

def run_benchmark(n_values: list[int], *, repeat: int, base_port: int, dealer_rank: int=0, startup_delay: float=2.0, timeout: float=300.0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    logger = _setup_logging()
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    port_cursor = 0
    for n in n_values:
        try:
            t = validate_party_count(n)
        except ValueError as exc:
            logger.error('invalid n=%s: %s', n, exc)
            for repeat_id in range(repeat):
                run = _failure_run(n, repeat_id)
                long_rows.extend(build_long_rows(run))
                wide_rows.append(build_wide_row(run))
            continue
        for repeat_id in range(repeat):
            run_base_port = base_port + port_cursor
            port_cursor += n + 32
            logger.info('VSS stage benchmark n=%s t=%s repeat=%s base_port=%s', n, t, repeat_id, run_base_port)
            try:
                run = run_single_vss_stage_benchmark(n, repeat_id=repeat_id, base_port=run_base_port, dealer_rank=dealer_rank, startup_delay=startup_delay, timeout=timeout)
                if not run['success']:
                    logger.error('VSS stage correctness failed n=%s t=%s repeat_id=%s secret=%s reconstructed=%s shares=%s', n, t, repeat_id, run['_secret'], run['_reconstructed'], run['_shares'])
                else:
                    lat = run['stage_latencies']
                    logger.info('VSS stage ok n=%s repeat=%s total_ms=%s dealer=%s com1=%s com2_dec=%s com2_bcast=%s com2_scalar=%s public=%s com3=%s', n, repeat_id, lat['total_time_ms'], lat['dealer_ms'], lat['com1_ms'], lat['com2_decode_ms'], lat['com2_broadcast_ms'], lat['com2_send_scalar_ms'], lat['public_ms'], lat['com3_ms'])
            except Exception as exc:
                logger.exception('VSS stage run failed n=%s t=%s repeat_id=%s: %s', n, t, repeat_id, exc)
                run = _failure_run(n, repeat_id, t=t)
            long_rows.extend(build_long_rows(run))
            wide_rows.append(build_wide_row(run))
    return (long_rows, wide_rows)

def main(argv: list[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description='Benchmark ΠVSS per-stage latency and logical communication')
    parser.add_argument('--n', type=str, default='5', help='comma-separated committee sizes')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=23000)
    parser.add_argument('--startup-delay', type=float, default=2.0)
    parser.add_argument('--timeout', type=float, default=300.0)
    parser.add_argument('--out-long', type=str, default='bench_output/vss_stage_by_n_long.csv')
    parser.add_argument('--out-wide', type=str, default='bench_output/vss_stage_by_n_wide.csv')
    args = parser.parse_args(argv)
    n_values = _parse_n_list(args.n)
    long_rows, wide_rows = run_benchmark(n_values, repeat=args.repeat, base_port=args.base_port, startup_delay=args.startup_delay, timeout=args.timeout)
    write_long_csv(args.out_long, long_rows)
    write_wide_csv(args.out_wide, wide_rows)
    print(f'wrote {args.out_long} ({len(long_rows)} rows), {args.out_wide} ({len(wide_rows)} rows)')
if __name__ == '__main__':
    main()