from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from cubempc.config import default_threshold, validate_party_count
from cubempc.field import mod, rand_field
from cubempc.metrics import SUBPROTOCOL_CSV_FIELDS, vss_logical_cost
from cubempc.protocols.vss import run_vss_multiprocess
PROTOCOL_NAME = 'VSS'
CSV_FIELDS = SUBPROTOCOL_CSV_FIELDS
vss_communication_cost = vss_logical_cost
LOG_PATH = Path('logs/vss_benchmark.log')
_PORT_STRIDE_MIN = 200

def next_vss_by_n_path(results_dir: Path | str='results') -> Path:
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    k = 1
    while True:
        path = root / f'vss_by_n{k}.csv'
        if not path.exists():
            return path
        k += 1

def _setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('vss_benchmark')
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

def run_single_vss_benchmark(n: int, *, repeat_id: int, base_port: int, dealer_rank: int=0, startup_delay: float=2.0, timeout: float=180.0, start_layer: int=0) -> dict[str, Any]:
    t = validate_party_count(n)
    secret = rand_field()
    instance_id = f'vss-n{n}-rep{repeat_id}'
    report = run_vss_multiprocess(n=n, secret=secret, t=t, base_port=base_port, start_layer=start_layer, dealer_rank=dealer_rank, instance_id=instance_id, startup_delay=startup_delay, timeout=timeout)
    shares = report['shares']
    reconstructed = report['reconstructed']
    secret_mod = mod(secret)
    success = reconstructed == secret_mod
    logical = vss_logical_cost(n, t)
    return {'protocol': PROTOCOL_NAME, 'n': n, 't': t, 'repeat_id': repeat_id, 'total_time_ms': round(float(report['total_time_ms']), 3), **logical, 'success': success, '_secret': secret_mod, '_reconstructed': reconstructed, '_shares': shares}

def _failure_row(n: int, repeat_id: int, *, t: int | None=None) -> dict[str, Any]:
    threshold = t if t is not None else default_threshold(n)
    logical = vss_logical_cost(n, threshold)
    return {'protocol': PROTOCOL_NAME, 'n': n, 't': threshold, 'repeat_id': repeat_id, 'total_time_ms': 0.0, **logical, 'success': False}

def write_vss_csv(path: str, rows: list[dict[str, Any]]) -> None:
    from cubempc.csv_io import write_aligned_csv
    write_aligned_csv(path, CSV_FIELDS, [{k: row[k] for k in CSV_FIELDS} for row in rows])

def run_benchmark(n_values: list[int], *, repeat: int, base_port: int, dealer_rank: int=0, startup_delay: float=2.5, timeout: float=600.0, cooldown: float=1.0) -> list[dict[str, Any]]:
    logger = _setup_logging()
    rows: list[dict[str, Any]] = []
    port_cursor = 0
    for n in n_values:
        try:
            t = validate_party_count(n)
        except ValueError as exc:
            logger.error('invalid n=%s: %s', n, exc)
            for rep in range(repeat):
                rows.append(_failure_row(n, rep))
            continue
        for repeat_id in range(repeat):
            run_base_port = base_port + port_cursor
            port_cursor += max(n + 32, _PORT_STRIDE_MIN)
            row = _failure_row(n, repeat_id, t=t)
            for attempt in range(2):
                attempt_port = run_base_port + attempt * _PORT_STRIDE_MIN * max(1, repeat)
                logger.info('VSS benchmark n=%s t=%s repeat=%s base_port=%s attempt=%s', n, t, repeat_id, attempt_port, attempt + 1)
                try:
                    row = run_single_vss_benchmark(n, repeat_id=repeat_id, base_port=attempt_port, dealer_rank=dealer_rank, startup_delay=startup_delay, timeout=timeout)
                    if not row['success']:
                        logger.error('VSS correctness failed n=%s t=%s repeat_id=%s secret=%s reconstructed=%s shares=%s', n, t, repeat_id, row['_secret'], row['_reconstructed'], row['_shares'])
                    else:
                        logger.info('VSS ok n=%s repeat=%s time_ms=%s logical_p2p=%s logical_bcast=%s logical_total=%s', n, repeat_id, row['total_time_ms'], row['logical_p2p_bytes'], row['logical_broadcast_bytes'], row['logical_total_bytes'])
                        break
                except Exception as exc:
                    logger.exception('VSS run failed n=%s t=%s repeat_id=%s attempt=%s: %s', n, t, repeat_id, attempt + 1, exc)
                    row = _failure_row(n, repeat_id, t=t)
                if cooldown > 0:
                    time.sleep(cooldown)
            rows.append({k: row[k] for k in CSV_FIELDS})
            if cooldown > 0:
                time.sleep(cooldown)
    return rows

def main(argv: list[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description='Benchmark ΠVSS by committee size n')
    parser.add_argument('--n', type=str, default='5', help='comma-separated committee sizes')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=None, help='base TCP port (default: 25000 + pid offset)')
    parser.add_argument('--startup-delay', type=float, default=2.5)
    parser.add_argument('--timeout', type=float, default=600.0)
    parser.add_argument('--cooldown', type=float, default=1.0, help='seconds between runs for port/process cleanup')
    parser.add_argument('--out', type=str, default=None, help='output CSV path (default: next bench_output/vss_by_n{k}.csv)')
    parser.add_argument('--results-dir', type=str, default='bench_output', help='directory for auto-numbered vss_by_n{k}.csv files')
    args = parser.parse_args(argv)
    out_path = Path(args.out) if args.out else next_vss_by_n_path(args.results_dir)
    base_port = args.base_port if args.base_port is not None else 25000 + os.getpid() % 8000
    n_values = _parse_n_list(args.n)
    rows = run_benchmark(n_values, repeat=args.repeat, base_port=base_port, startup_delay=args.startup_delay, timeout=args.timeout, cooldown=args.cooldown)
    write_vss_csv(str(out_path), rows)
    print(f'wrote {out_path} ({len(rows)} rows)')
if __name__ == '__main__':
    main()