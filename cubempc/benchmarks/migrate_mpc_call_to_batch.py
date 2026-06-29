from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from cubempc.benchmarks.run_mpc_stage_raw import BATCH_FIELDS, MUL_MASK_STAGES, MUL_ONLINE_STAGES, TaskSample, VSS_STAGE_NAMES, _aggregate_batch_rows

def _float(value: str | None) -> float:
    if value is None or value == '':
        return 0.0
    return float(value)

def _call_row_to_samples(row: dict[str, str]) -> list[TaskSample]:
    call_type = row['call_type']
    rank = int(row['rank'])
    layer_id = int(row['layer_id'])
    gate_id = int(row['gate_id'])
    vss_round_id = int(row['vss_round_id'])
    n = int(row['n'])
    samples: list[TaskSample] = []
    if call_type == 'INPUT_VSS':
        for stage, _ in VSS_STAGE_NAMES:
            samples.append(TaskSample(phase='INPUT_VSS', layer_id=-1, stage=stage, task_id=gate_id, rank=rank, cpu_ms=_float(row.get(f'{stage}_cpu_ms')), wall_ms=_float(row.get(f'{stage}_wall_ms'))))
        return samples
    if call_type == 'RG_FOR_MUL':
        task_id = gate_id * n + vss_round_id
        for stage, _ in VSS_STAGE_NAMES:
            samples.append(TaskSample(phase='LAYER_RG_PREP', layer_id=layer_id, stage=stage, task_id=task_id, rank=rank, cpu_ms=_float(row.get(f'{stage}_cpu_ms')), wall_ms=_float(row.get(f'{stage}_wall_ms'))))
        if vss_round_id == n - 1:
            samples.append(TaskSample(phase='LAYER_RG_PREP', layer_id=layer_id, stage='rg_combine', task_id=gate_id, rank=rank, cpu_ms=_float(row.get('rg_combine_cpu_ms')), wall_ms=_float(row.get('rg_combine_wall_ms'))))
        return samples
    if call_type == 'MUL_MASK_PREP':
        for stage in MUL_MASK_STAGES:
            samples.append(TaskSample(phase='LAYER_MUL_MASK_PREP', layer_id=layer_id, stage=stage, task_id=gate_id, rank=rank, cpu_ms=_float(row.get(f'{stage}_cpu_ms')), wall_ms=_float(row.get(f'{stage}_wall_ms'))))
        return samples
    if call_type == 'MUL_ONLINE':
        for stage in MUL_ONLINE_STAGES:
            samples.append(TaskSample(phase='LAYER_MUL_ONLINE', layer_id=layer_id, stage=stage, task_id=gate_id, rank=rank, cpu_ms=_float(row.get(f'{stage}_cpu_ms')), wall_ms=_float(row.get(f'{stage}_wall_ms'))))
        return samples
    return samples

def _load_count_meta(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        return {(row['n'], row['d'], row['repeat_id']): row for row in reader}

def migrate(*, in_calls: Path, in_counts: Path, out_batch: Path) -> int:
    count_meta = _load_count_meta(in_counts)
    grouped_rows: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with in_calls.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row['n'], row['d'], row['repeat_id'])
            if key not in count_meta:
                continue
            grouped_rows[key].append(row)
    batch_rows: list[dict[str, object]] = []
    for key in sorted(grouped_rows.keys(), key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
        rows = grouped_rows[key]
        meta = count_meta[key]
        first = rows[0]
        samples: list[TaskSample] = []
        for row in rows:
            samples.extend(_call_row_to_samples(row))
        success = meta.get('success', '').lower() == 'true'
        batch_rows.extend(_aggregate_batch_rows(scheme=first['scheme'], n=int(first['n']), t=int(first['t']), d=int(first['d']), w=int(first['w']), num_input_secrets=int(first['num_input_secrets']), repeat_id=int(first['repeat_id']), success=success, error=meta.get('error', ''), samples=samples if success else []))
    out_batch.parent.mkdir(parents=True, exist_ok=True)
    with out_batch.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=BATCH_FIELDS)
        writer.writeheader()
        writer.writerows(batch_rows)
    return len(batch_rows)

def main() -> None:
    parser = argparse.ArgumentParser(description='Migrate legacy MPC call CSV to batch CSV')
    parser.add_argument('--in-calls', type=Path, default=Path('bench_output/mpc_raw/mpc_call_stage_raw.csv'))
    parser.add_argument('--in-counts', type=Path, default=Path('bench_output/mpc_raw/mpc_count_raw.csv'))
    parser.add_argument('--out-batch', type=Path, default=Path('bench_output/mpc_raw/mpc_batch_stage_raw.csv'))
    args = parser.parse_args()
    rows = migrate(in_calls=args.in_calls, in_counts=args.in_counts, out_batch=args.out_batch)
    print(f'wrote {args.out_batch} ({rows} rows)')
if __name__ == '__main__':
    main()