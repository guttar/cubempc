from __future__ import annotations
import argparse
import math
import socket
from pathlib import Path
from random import Random
from typing import Any
from cubempc.circuits.circuit import Circuit, count_gates_by_op
from cubempc.circuits.generator import make_synthetic_layered_circuit
from cubempc.config import validate_party_count
from cubempc.csv_io import read_aligned_csv, write_aligned_csv
from cubempc.field import rand_field
from cubempc.protocols.mpc import run_cubempc_multiprocess
RAW_FIELDS = ['n', 't', 'num_inputs', 'depth', 'width', 'mul_ratio', 'mul_gates', 'repeat_id', 'randomness_mode', 'latency_ms', 'p2p_bytes', 'broadcast_bytes', 'total_bytes', 'message_count']

def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(',') if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError('expected at least one integer')
    return values

def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(',') if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError('expected at least one float')
    return values

def _free_port_block(n: int) -> int:
    for _ in range(200):
        probe = socket.socket()
        probe.bind(('127.0.0.1', 0))
        base = probe.getsockname()[1]
        probe.close()
        if base + n >= 65535:
            continue
        sockets = []
        try:
            for port in range(base, base + n):
                sock = socket.socket()
                sock.bind(('127.0.0.1', port))
                sockets.append(sock)
            return base
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError(f'could not allocate {n} consecutive free ports')

def _circuit_stats(circuit: Circuit) -> dict[str, int]:
    counts = count_gates_by_op(circuit)
    return {'num_inputs': len(circuit.inputs), 'mul_gates': counts['mul']}

def _make_inputs(circuit: Circuit, seed: int, n: int) -> tuple[dict[str, int], dict[str, int]]:
    rng = Random(seed)
    input_values = {wire: rand_field(rng) for wire in circuit.inputs}
    input_dealers = {wire: idx % n for idx, wire in enumerate(circuit.inputs)}
    return (input_values, input_dealers)

def _write_csv(path: str, fields: list[str], rows: list[dict[str, Any]]) -> None:
    write_aligned_csv(path, fields, rows)

def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (int(row['n']), int(row['t']), int(row['num_inputs']), int(row['depth']), int(row['width']), float(row['mul_ratio']), int(row['repeat_id']), row['randomness_mode'])

def _load_existing_rows(path: str) -> list[dict[str, Any]]:
    raw_rows = read_aligned_csv(path, RAW_FIELDS)
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        rows.append({**row, 'n': int(row['n']), 't': int(row['t']), 'num_inputs': int(row['num_inputs']), 'depth': int(row['depth']), 'width': int(row['width']), 'mul_ratio': float(row['mul_ratio']), 'mul_gates': int(row['mul_gates']), 'repeat_id': int(row['repeat_id']), 'latency_ms': float(row['latency_ms']), 'p2p_bytes': int(row['p2p_bytes']), 'broadcast_bytes': int(row['broadcast_bytes']), 'total_bytes': int(row['total_bytes']), 'message_count': int(row['message_count']), 'success': str(row.get('success', 'true')).lower() in {'1', 'true', 'yes'}})
    return rows

def _persist_results(out_path: str, rows: list[dict[str, Any]]) -> None:
    _write_csv(out_path, RAW_FIELDS, rows)

def _run_one(*, n: int, t: int, num_inputs: int, depth: int, width: int, mul_ratio: float, repeat_id: int, seed: int, startup_delay: float, timeout: float, randomness_mode: str) -> dict[str, Any]:
    circuit = make_synthetic_layered_circuit(depth=depth, width=width, mul_ratio=mul_ratio, seed=seed, num_inputs=num_inputs)
    stats = _circuit_stats(circuit)
    base = {'n': n, 't': t, 'depth': depth, 'width': width, 'mul_ratio': mul_ratio, 'repeat_id': repeat_id, 'randomness_mode': randomness_mode, **stats, 'latency_ms': 0.0, 'p2p_bytes': 0, 'broadcast_bytes': 0, 'total_bytes': 0, 'message_count': 0, 'success': False, 'error': ''}
    try:
        input_values, input_dealers = _make_inputs(circuit, seed + 1000003, n)
        result = run_cubempc_multiprocess(circuit, input_values, input_dealers, n=n, t=t, base_port=_free_port_block(n), instance_id=f'mpc-d{depth}-w{width}-r{repeat_id}-s{seed}', startup_delay=startup_delay, timeout=timeout, randomness_mode=randomness_mode)
        logical = result.get('logical_metrics', {})
        base.update({'latency_ms': round(float(result.get('protocol_latency_ms', 0.0)), 3), 'p2p_bytes': int(logical.get('logical_p2p_bytes', 0)), 'broadcast_bytes': int(logical.get('logical_broadcast_bytes', 0)), 'total_bytes': int(logical.get('logical_total_bytes', 0)), 'message_count': int(logical.get('logical_message_count', 0)), 'success': bool(result.get('match', False)), 'error': '' if result.get('match', False) else 'reconstructed output mismatch'})
    except Exception as exc:
        base['error'] = f'{type(exc).__name__}: {exc}'
    return base

def main(argv: list[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description='Benchmark full CUBE MPC')
    parser.add_argument('--n', type=_parse_int_list, default=[5])
    parser.add_argument('--depth', type=_parse_int_list, default=[2])
    parser.add_argument('--width', type=_parse_int_list, default=[10])
    parser.add_argument('--num-inputs', type=_parse_int_list, default=None)
    parser.add_argument('--mul-ratio', type=_parse_float_list, default=[0.5])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--randomness-mode', choices=['local', 'rg'], default='rg')
    parser.add_argument('--out', '--output', dest='out', type=str, default='bench_output/bench_mpc.csv')
    parser.add_argument('--startup-delay', type=float, default=3.5)
    parser.add_argument('--timeout', type=float, default=600.0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--resume', action='store_true', help='skip runs already present in the output CSV')
    args = parser.parse_args(argv)
    if args.repeat < 1:
        raise ValueError('--repeat must be >= 1')
    rows: list[dict[str, Any]] = _load_existing_rows(args.out) if args.resume else []
    existing_keys = {_row_key(row) for row in rows}
    for n in args.n:
        t = validate_party_count(n, None)
        for depth in args.depth:
            if depth < 1:
                raise ValueError(f'depth must be >= 1, got {depth}')
            for width in args.width:
                if width < 1:
                    raise ValueError(f'width must be >= 1, got {width}')
                num_inputs_values = args.num_inputs if args.num_inputs is not None else [width]
                for num_inputs in num_inputs_values:
                    if num_inputs < 1:
                        raise ValueError(f'num_inputs must be >= 1, got {num_inputs}')
                    for mul_ratio in args.mul_ratio:
                        if not 0.0 <= mul_ratio <= 1.0:
                            raise ValueError(f'mul_ratio must be in [0, 1], got {mul_ratio}')
                        for repeat_id in range(args.repeat):
                            run_seed = int(args.seed + n * 1000000 + num_inputs * 100000 + depth * 10000 + width * 100 + math.floor(mul_ratio * 10000) + repeat_id)
                            pending_key = (n, t, num_inputs, depth, width, mul_ratio, repeat_id, args.randomness_mode)
                            if pending_key in existing_keys:
                                print(f'MPC n={n} t={t} inputs={num_inputs} depth={depth} width={width} mul_ratio={mul_ratio} randomness={args.randomness_mode} repeat={repeat_id}: skipped (already in CSV)')
                                continue
                            row = _run_one(n=n, t=t, num_inputs=num_inputs, depth=depth, width=width, mul_ratio=mul_ratio, repeat_id=repeat_id, seed=run_seed, startup_delay=args.startup_delay, timeout=args.timeout, randomness_mode=args.randomness_mode)
                            status = 'ok' if row['success'] else 'failed'
                            if row['success']:
                                rows.append(row)
                                existing_keys.add(_row_key(row))
                                _persist_results(args.out, rows)
                            print(f'MPC n={n} t={t} inputs={num_inputs} depth={depth} width={width} mul_ratio={mul_ratio} randomness={args.randomness_mode} repeat={repeat_id}: {status}')
                            print(f'wrote {args.out} ({len(rows)} rows)')
    if rows:
        _persist_results(args.out, rows)
        print(f'wrote {args.out}')
if __name__ == '__main__':
    main()