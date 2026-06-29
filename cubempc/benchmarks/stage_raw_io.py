from __future__ import annotations
import csv
import gzip
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol
STAGE_TASK_RAW_FIELDS = ['scheme', 'protocol', 'n', 't', 'd', 'w', 'num_input_secrets', 'repeat_id', 'phase', 'layer_id', 'stage', 'rank_id', 'task_id', 'task_count', 'cpu_ms', 'wall_ms', 'p2p_bytes', 'broadcast_bytes', 'total_bytes', 'message_count', 'success', 'error']
STAGE_RANK_RAW_FIELDS = STAGE_TASK_RAW_FIELDS
STAGE_RANK_AGG_FIELDS = ['scheme', 'protocol', 'n', 't', 'd', 'w', 'num_input_secrets', 'repeat_id', 'phase', 'layer_id', 'stage', 'rank_id', 'task_count', 'cpu_ms', 'wall_ms', 'p2p_bytes', 'broadcast_bytes', 'total_bytes', 'message_count', 'success', 'error']
STAGE_SUMMARY_FIELDS = ['scheme', 'n', 't', 'd', 'w', 'repeat_id', 'phase', 'layer_id', 'stage', 'task_count', 'cpu_sum_ms', 'cpu_p50_ms', 'cpu_p90_ms', 'cpu_max_ms', 'wall_max_ms', 'p2p_bytes', 'broadcast_bytes', 'total_bytes', 'message_count', 'success', 'error']

@dataclass(frozen=True)
class StageComm:
    p2p_bytes: int
    broadcast_bytes: int
    message_count: int

    @property
    def total_bytes(self) -> int:
        return self.p2p_bytes + self.broadcast_bytes

class StageSample(Protocol):
    phase: str
    layer_id: int
    stage: str
    task_id: int
    rank: int
    cpu_ms: float
    wall_ms: float

def _ms(value: float) -> float:
    return round(float(value), 6)

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

def _vss_stage_weights(n: int, t: int, field_bytes: int) -> dict[str, float]:
    dealer = n * (t + 1) ** 2 * field_bytes
    com1 = n * n * (t + 1) * field_bytes
    com2 = n * n * n * field_bytes
    broadcast = (n * n + 7) // 8
    weights = {'vss_dealer': dealer, 'vss_com1': com1, 'vss_com2_decode': 0.0, 'vss_com2_broadcast': broadcast, 'vss_com2_send': com2, 'vss_public': 0.0, 'vss_com3': 0.0}
    return weights

def _allocate_by_weights(stage_names: Iterable[str], weights: dict[str, float], *, p2p_total: int, broadcast_total: int, message_total: int) -> dict[str, StageComm]:
    p2p_weight_sum = sum((weights.get(stage, 0.0) for stage in stage_names))
    broadcast_weight_sum = sum((weights.get(stage, 0.0) for stage in stage_names if stage.endswith('_broadcast') or stage == 'vss_com2_broadcast'))
    if broadcast_weight_sum == 0.0:
        broadcast_weight_sum = float(len(stage_names))
    out: dict[str, StageComm] = {}
    p2p_allocated = 0
    broadcast_allocated = 0
    names = list(stage_names)
    for idx, stage in enumerate(names):
        if idx == len(names) - 1:
            p2p_bytes = p2p_total - p2p_allocated
            broadcast_bytes = broadcast_total - broadcast_allocated
        else:
            if p2p_weight_sum > 0:
                p2p_bytes = int(p2p_total * weights.get(stage, 0.0) / p2p_weight_sum)
            else:
                p2p_bytes = p2p_total // len(names)
            if broadcast_total > 0:
                if stage.endswith('_broadcast') or stage == 'vss_com2_broadcast':
                    broadcast_bytes = int(broadcast_total * weights.get(stage, 0.0) / broadcast_weight_sum)
                else:
                    broadcast_bytes = 0
            else:
                broadcast_bytes = 0
        p2p_allocated += p2p_bytes
        broadcast_allocated += broadcast_bytes
        msg = message_total // len(names) if message_total else 0
        out[stage] = StageComm(p2p_bytes=p2p_bytes, broadcast_bytes=broadcast_bytes, message_count=msg)
    if names and message_total:
        out[names[-1]] = StageComm(p2p_bytes=out[names[-1]].p2p_bytes, broadcast_bytes=out[names[-1]].broadcast_bytes, message_count=message_total - msg * (len(names) - 1))
    return out

def cube_stage_comm_map(*, n: int, t: int, d: int, w: int, num_input_secrets: int, field_bytes: int, rg_mode: str, input_vss_stages: list[str], rg_prep_stages: list[str], mul_mask_stages: list[str], mul_online_stages: list[str], comm_row: dict[str, Any]) -> dict[tuple[str, int, str], StageComm]:
    vss_weights = _vss_stage_weights(n, t, field_bytes)
    vss_message_count = 2 * n * n + 2 * n
    out: dict[tuple[str, int, str], StageComm] = {}
    input_alloc = _allocate_by_weights(input_vss_stages, vss_weights, p2p_total=int(comm_row['input_vss_p2p_bytes']), broadcast_total=int(comm_row['input_vss_broadcast_bytes']), message_total=num_input_secrets * vss_message_count)
    for stage, comm in input_alloc.items():
        out['INPUT_VSS', -1, stage] = comm
    num_rg_vss = d * n if rg_mode == 'batch_layer' else d * w * n
    rg_scale = w if rg_mode == 'batch_layer' else 1
    rg_alloc = _allocate_by_weights(rg_prep_stages, vss_weights, p2p_total=int(comm_row['rg_p2p_bytes']), broadcast_total=int(comm_row['rg_broadcast_bytes']), message_total=num_rg_vss * vss_message_count)
    for layer_id in range(d):
        for stage, comm in rg_alloc.items():
            if stage == 'rg_combine':
                out['LAYER_RG_PREP', layer_id, stage] = StageComm(0, 0, 0)
            else:
                scaled = StageComm(p2p_bytes=int(comm.p2p_bytes / d / max(rg_scale, 1)), broadcast_bytes=int(comm.broadcast_bytes / d / max(rg_scale, 1)), message_count=max(comm.message_count // d, 0))
                out['LAYER_RG_PREP', layer_id, stage] = scaled
    mul_mask_map = {'mul_mask_compute': StageComm(0, 0, 0), 'mul_mask_send': StageComm(int(comm_row['mul_mask_p2p_bytes'] / d), 0, d * w * n * n)}
    for layer_id in range(d):
        for stage, comm in mul_mask_map.items():
            out['LAYER_MUL_MASK_PREP', layer_id, stage] = StageComm(p2p_bytes=comm.p2p_bytes // max(w, 1) if stage == 'mul_mask_send' else 0, broadcast_bytes=0, message_count=max(comm.message_count // (d * max(w, 1)), 0))
    mul_online_map = {'mul_mask_reconstruct': StageComm(0, 0, 0), 'mul_product_compute': StageComm(0, 0, 0), 'mul_product_broadcast': StageComm(0, int(comm_row['mul_online_broadcast_bytes'] / d), 0), 'mul_public_decode': StageComm(0, 0, 0), 'mul_rs_construct': StageComm(0, 0, 0), 'mul_rs_send': StageComm(int(comm_row['rs_p2p_bytes'] / d), 0, 0), 'mul_rs_decode': StageComm(0, 0, 0), 'mul_output_add': StageComm(int(comm_row['mul_online_p2p_bytes'] / d), 0, d * w * n * n)}
    for layer_id in range(d):
        for stage, comm in mul_online_map.items():
            per_gate = StageComm(p2p_bytes=comm.p2p_bytes // max(w, 1), broadcast_bytes=comm.broadcast_bytes // max(w, 1), message_count=max(comm.message_count // (d * max(w, 1)), 0))
            out['LAYER_MUL_ONLINE', layer_id, stage] = per_gate
    out['OUTPUT_RECON', d, 'output_recon'] = StageComm(p2p_bytes=int(comm_row['output_recon_p2p_bytes']), broadcast_bytes=int(comm_row['output_recon_broadcast_bytes']), message_count=0)
    return out

def build_stage_rank_rows(*, scheme: str, protocol: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, success: bool, error: str, samples: Iterable[StageSample], stage_comm: dict[tuple[str, int, str], StageComm], stage_task_counts: dict[tuple[str, int, str], int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        key = (sample.phase, sample.layer_id, sample.stage)
        comm = stage_comm.get(key, StageComm(0, 0, 0))
        task_count = stage_task_counts.get(key, 1)
        rows.append({'scheme': scheme, 'protocol': protocol, 'n': n, 't': t, 'd': d, 'w': w, 'num_input_secrets': num_input_secrets, 'repeat_id': repeat_id, 'phase': sample.phase, 'layer_id': sample.layer_id, 'stage': sample.stage, 'rank_id': sample.rank, 'task_id': sample.task_id, 'task_count': task_count, 'cpu_ms': _ms(sample.cpu_ms), 'wall_ms': _ms(sample.wall_ms), 'p2p_bytes': comm.p2p_bytes, 'broadcast_bytes': comm.broadcast_bytes, 'total_bytes': comm.total_bytes, 'message_count': comm.message_count, 'success': 'true' if success else 'false', 'error': error})
    return rows

def _rank_agg_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row['scheme'], row['protocol'], row['n'], row['t'], row['d'], row['w'], row['num_input_secrets'], row['repeat_id'], row['phase'], row['layer_id'], row['stage'], row['rank_id'])

def aggregate_stage_rank_rows(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[_rank_agg_key(row)].append(row)
    agg_rows: list[dict[str, Any]] = []
    for rows in grouped.values():
        first = rows[0]
        task_ids = {row['task_id'] for row in rows}
        successes = [str(row.get('success', '')).lower() == 'true' for row in rows]
        errors = [str(row.get('error', '')).strip() for row in rows if str(row.get('error', '')).strip()]
        agg_rows.append({'scheme': first['scheme'], 'protocol': first['protocol'], 'n': first['n'], 't': first['t'], 'd': first['d'], 'w': first['w'], 'num_input_secrets': first['num_input_secrets'], 'repeat_id': first['repeat_id'], 'phase': first['phase'], 'layer_id': first['layer_id'], 'stage': first['stage'], 'rank_id': first['rank_id'], 'task_count': len(task_ids) if task_ids else len(rows), 'cpu_ms': _ms(sum((float(row['cpu_ms']) for row in rows))), 'wall_ms': _ms(max((float(row['wall_ms']) for row in rows))), 'p2p_bytes': int(first['p2p_bytes']), 'broadcast_bytes': int(first['broadcast_bytes']), 'total_bytes': int(first['total_bytes']), 'message_count': int(first['message_count']), 'success': 'true' if all(successes) else 'false', 'error': ' | '.join(errors)})
    return agg_rows

def aggregate_stage_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row['scheme'], row['n'], row['t'], row['d'], row['w'], row['repeat_id'], row['phase'], row['layer_id'], row['stage'])
        grouped[key].append(row)
    summary_rows: list[dict[str, Any]] = []
    for rows in grouped.values():
        cpu_values = [float(r['cpu_ms']) for r in rows]
        wall_values = [float(r['wall_ms']) for r in rows]
        task_counts = [int(r['task_count']) for r in rows if str(r.get('task_count', '')).strip()]
        first = rows[0]
        summary_rows.append({'scheme': first['scheme'], 'n': first['n'], 't': first['t'], 'd': first['d'], 'w': first['w'], 'repeat_id': first['repeat_id'], 'phase': first['phase'], 'layer_id': first['layer_id'], 'stage': first['stage'], 'task_count': max(task_counts) if task_counts else int(first['task_count']), 'cpu_sum_ms': _ms(sum(cpu_values)), 'cpu_p50_ms': _ms(statistics.median(cpu_values)), 'cpu_p90_ms': _ms(_percentile(cpu_values, 90.0)), 'cpu_max_ms': _ms(max(cpu_values) if cpu_values else 0.0), 'wall_max_ms': _ms(max(wall_values) if wall_values else 0.0), 'p2p_bytes': first['p2p_bytes'], 'broadcast_bytes': first['broadcast_bytes'], 'total_bytes': first['total_bytes'], 'message_count': first['message_count'], 'success': first['success'], 'error': first['error']})
    return summary_rows

def build_stage_outputs(*, scheme: str, protocol: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, success: bool, error: str, samples: Iterable[StageSample], stage_comm: dict[tuple[str, int, str], StageComm], stage_task_counts: dict[tuple[str, int, str], int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows = build_stage_rank_rows(scheme=scheme, protocol=protocol, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, success=success, error=error, samples=samples, stage_comm=stage_comm, stage_task_counts=stage_task_counts)
    rank_agg_rows = aggregate_stage_rank_rows(task_rows)
    summary_rows = aggregate_stage_summary_rows(rank_agg_rows)
    return (task_rows, rank_agg_rows, summary_rows)

def write_csv_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})

def append_csv_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})

def append_gz_csv_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with gzip.open(path, 'at', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})