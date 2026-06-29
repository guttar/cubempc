from __future__ import annotations
import argparse
import asyncio
import csv
import hashlib
import multiprocessing as mp
import pickle
import socket
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from random import Random
from typing import Any
from cubempc.async_util import bounded_gather
from cubempc.algebra_cache import get_cache_stats, reset_cache_stats
from cubempc.benchmarks.distributed import distributed_layout, launch_remote_rank, load_worker_result, wait_for_result_files
from cubempc.benchmarks.stage_raw_io import STAGE_RANK_AGG_FIELDS, STAGE_SUMMARY_FIELDS, STAGE_TASK_RAW_FIELDS, append_gz_csv_rows, build_stage_outputs, cube_stage_comm_map
from cubempc.host_layout import HostLayout
from cubempc.profiling import get_counters, reset_counters
from cubempc.vss_profiling import VssProfiler, set_profiler
from cubempc.messages import TRANSPORT_JSON, TRANSPORT_OBJECT
from cubempc.config import MPCConfig, validate_party_count
from cubempc.field import mod, rand_field
from cubempc.protocols.mpc import reconstruct_output
from cubempc.protocols.multiplication import resharing_prep_keys, run_mul
from cubempc.protocols.rg import combine_vandermonde_shares, map_rg_output_to_mul_keys, run_rg_batch
from cubempc.protocols.vss import _ps, _share_key, party_id, run_vss
from cubempc.shamir import shamir_share
PROTOCOL = 'MPC'
DEFAULT_SCHEME = 'CUBE'
RG_MODE_PER_GATE = 'per_gate'
RG_MODE_BATCH_LAYER = 'batch_layer'
DEFAULT_RG_MODE = RG_MODE_BATCH_LAYER
PORT_STRIDE = 1000
NUM_MACHINES = 6
VSS_STAGE_NAMES = (('vss_dealer', 'dealer'), ('vss_com1', 'com1'), ('vss_com2_decode', 'com2_decode'), ('vss_com2_broadcast', 'com2_broadcast'), ('vss_com2_send', 'com2_send'), ('vss_public', 'public'), ('vss_com3', 'com3'))
INPUT_VSS_STAGES = [name for name, _ in VSS_STAGE_NAMES]
RG_PREP_STAGES = [*INPUT_VSS_STAGES, 'rg_combine']
MUL_MASK_STAGES = ['mul_mask_compute', 'mul_mask_send']
MUL_ONLINE_STAGES = ['mul_mask_reconstruct', 'mul_product_compute', 'mul_product_broadcast', 'mul_public_decode', 'mul_rs_construct', 'mul_rs_send', 'mul_rs_decode', 'mul_output_add']
CPU_FIELDS = ['vss_dealer_cpu_ms', 'vss_com1_cpu_ms', 'vss_com2_decode_cpu_ms', 'vss_com2_broadcast_cpu_ms', 'vss_com2_send_cpu_ms', 'vss_public_cpu_ms', 'vss_com3_cpu_ms', 'rg_combine_cpu_ms', 'mul_mask_compute_cpu_ms', 'mul_mask_send_cpu_ms', 'mul_mask_reconstruct_cpu_ms', 'mul_product_compute_cpu_ms', 'mul_product_broadcast_cpu_ms', 'mul_public_decode_cpu_ms', 'mul_rs_construct_cpu_ms', 'mul_rs_send_cpu_ms', 'mul_rs_decode_cpu_ms', 'mul_output_add_cpu_ms', 'rs_construct_cpu_ms', 'rs_send_cpu_ms', 'rs_decode_cpu_ms', 'rs_output_cpu_ms', 'output_recon_cpu_ms']
WALL_FIELDS = ['vss_dealer_wall_ms', 'vss_com1_wall_ms', 'vss_com2_decode_wall_ms', 'vss_com2_broadcast_wall_ms', 'vss_com2_send_wall_ms', 'vss_public_wall_ms', 'vss_com3_wall_ms', 'rg_combine_wall_ms', 'mul_mask_compute_wall_ms', 'mul_mask_send_wall_ms', 'mul_mask_reconstruct_wall_ms', 'mul_product_compute_wall_ms', 'mul_product_broadcast_wall_ms', 'mul_public_decode_wall_ms', 'mul_rs_construct_wall_ms', 'mul_rs_send_wall_ms', 'mul_rs_decode_wall_ms', 'mul_output_add_wall_ms', 'rs_construct_wall_ms', 'rs_send_wall_ms', 'rs_decode_wall_ms', 'rs_output_wall_ms', 'output_recon_wall_ms']
CALL_FIELDS = ['scheme', 'protocol', 'n', 't', 'd', 'w', 'num_input_secrets', 'repeat_id', 'rank', 'call_type', 'layer_id', 'gate_id', 'vss_round_id', 'success', 'error', *CPU_FIELDS, *WALL_FIELDS, 'local_call_wall_ms']
BATCH_FIELDS = ['scheme', 'protocol', 'n', 't', 'd', 'w', 'num_input_secrets', 'repeat_id', 'phase', 'layer_id', 'stage', 'task_count', 'machine0_cpu_sum_ms', 'machine1_cpu_sum_ms', 'machine2_cpu_sum_ms', 'machine3_cpu_sum_ms', 'machine4_cpu_sum_ms', 'machine5_cpu_sum_ms', 'machine0_cpu_max_ms', 'machine1_cpu_max_ms', 'machine2_cpu_max_ms', 'machine3_cpu_max_ms', 'machine4_cpu_max_ms', 'machine5_cpu_max_ms', 'wall_max_ms', 'success', 'error']
COUNT_FIELDS = ['scheme', 'protocol', 'n', 't', 'd', 'w', 'num_input_secrets', 'repeat_id', 'rg_mode', 'success', 'error', 'num_input_vss', 'num_mul_gates', 'num_rg_calls', 'num_rg_batch_calls', 'num_rg_outputs', 'num_mul_mask_prep_calls', 'num_mul_online_calls', 'num_rs_calls', 'num_output_recon', 'input_vss_ms', 'rg_prep_ms', 'mul_mask_prep_ms', 'mul_online_ms', 'output_recon_ms', 'total_wall_ms', 'critical_path_ms', 'total_cpu_ms', 'local_total_wall_ms', 'rg_vss_instance_count']
VSS_PROFILE_FIELDS = ['scheme', 'n', 't', 'd', 'w', 'repeat_id', 'phase', 'layer_id', 'stage', 'rank_id', 'op_name', 'op_count', 'cpu_ms', 'wall_ms']
RG_COUNTER_FIELDS = ['scheme', 'n', 't', 'd', 'w', 'num_input_secrets', 'repeat_id', 'rg_mode', 'rg_call_count', 'rg_batch_count', 'rg_output_count', 'rg_vss_instance_count', 'rg_vss_dealer_count', 'rg_vss_secret_count', 'vss_poly_sample_count', 'vss_encode_count', 'vss_decode_count', 'vss_broadcast_msg_count', 'vss_send_scalar_count', 'vss_reconstruct_count', 'lagrange_eval_count', 'bw_decode_count', 'fast_decode_count', 'fallback_decode_count', 'fast_path_count', 'correction_triggered_count', 'cache_hit_count', 'cache_miss_count']
COMM_FIELDS = ['scheme', 'protocol', 'n', 't', 'd', 'w', 'num_input_secrets', 'repeat_id', 'rg_mode', 'input_vss_p2p_bytes', 'input_vss_broadcast_bytes', 'rg_p2p_bytes', 'rg_broadcast_bytes', 'mul_mask_p2p_bytes', 'mul_online_p2p_bytes', 'mul_online_broadcast_bytes', 'rs_p2p_bytes', 'rs_broadcast_bytes', 'output_recon_p2p_bytes', 'output_recon_broadcast_bytes', 'total_p2p_bytes', 'total_broadcast_bytes', 'total_bytes', 'message_count']

@dataclass
class WireRef:
    layer: int
    key: Any

@dataclass
class TaskSample:
    phase: str
    layer_id: int
    stage: str
    task_id: int
    rank: int
    cpu_ms: float
    wall_ms: float

@dataclass
class RankRawResult:
    rank: int
    output_share: int
    local_total_wall_ms: float
    task_samples: list[TaskSample] = field(default_factory=list)
    call_csv_path: str = ''
    profile_counters: dict[str, int] = field(default_factory=dict)
    vss_profile_samples: list[dict[str, Any]] = field(default_factory=list)

def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(',') if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError('expected at least one integer')
    return values

def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'expected boolean, got {raw!r}')

def _ms(value: Any) -> float:
    return round(float(value or 0.0), 6)

def _seed(*parts: Any) -> int:
    text = ':'.join((str(part) for part in parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')

def _base_call_row(*, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, rank: int, call_type: str, layer_id: int, gate_id: int, vss_round_id: int, success: bool=True, error: str='') -> dict[str, Any]:
    row: dict[str, Any] = {'scheme': scheme, 'protocol': PROTOCOL, 'n': n, 't': t, 'd': d, 'w': w, 'num_input_secrets': num_input_secrets, 'repeat_id': repeat_id, 'rank': rank, 'call_type': call_type, 'layer_id': layer_id, 'gate_id': gate_id, 'vss_round_id': vss_round_id, 'success': 'true' if success else 'false', 'error': error, 'local_call_wall_ms': 0.0}
    for field_name in CPU_FIELDS + WALL_FIELDS:
        row[field_name] = 0.0
    return row

def _copy_vss_stage_metrics(row: dict[str, Any], stages: dict[str, float]) -> None:
    for out_prefix, source in VSS_STAGE_NAMES:
        row[f'{out_prefix}_cpu_ms'] = _ms(stages.get(f'{source}_cpu_ms', 0.0))
        row[f'{out_prefix}_wall_ms'] = _ms(stages.get(f'{source}_wall_ms', 0.0))

def _copy_timing(row: dict[str, Any], timing: dict[str, float], fields: list[str]) -> None:
    for field_name in fields:
        row[field_name] = _ms(timing.get(field_name, 0.0))

def _record_vss_tasks(recorder: list[TaskSample], *, phase: str, layer_id: int, task_id: int, rank: int, stages: dict[str, float], total_wall_ms: float) -> None:
    for stage, source in VSS_STAGE_NAMES:
        recorder.append(TaskSample(phase=phase, layer_id=layer_id, stage=stage, task_id=task_id, rank=rank, cpu_ms=float(stages.get(f'{source}_cpu_ms', 0.0)), wall_ms=float(stages.get(f'{source}_wall_ms', 0.0))))
    _ = total_wall_ms

def _record_task(recorder: list[TaskSample], *, phase: str, layer_id: int, stage: str, task_id: int, rank: int, cpu_ms: float, wall_ms: float) -> None:
    recorder.append(TaskSample(phase=phase, layer_id=layer_id, stage=stage, task_id=task_id, rank=rank, cpu_ms=cpu_ms, wall_ms=wall_ms))

def _expected_task_count(*, phase: str, layer_id: int, stage: str, num_input_secrets: int, d: int, w: int, n: int, rg_mode: str=DEFAULT_RG_MODE) -> int:
    if phase == 'INPUT_VSS':
        return num_input_secrets
    if phase == 'LAYER_RG_PREP':
        if stage == 'rg_combine':
            return w
        if rg_mode == RG_MODE_BATCH_LAYER:
            return n
        return w * n
    if phase in {'LAYER_MUL_MASK_PREP', 'LAYER_MUL_ONLINE'}:
        return w
    if phase == 'OUTPUT_RECON':
        return 0
    _ = (layer_id, d)
    raise ValueError(f'unknown phase {phase!r}')

def _phase_layer_wall_ms(samples: list[TaskSample], phase: str, layer_id: int) -> float:
    by_stage: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if sample.phase == phase and sample.layer_id == layer_id:
            by_stage[sample.stage].append(sample.wall_ms)
    if not by_stage:
        return 0.0
    return sum((max(values) for values in by_stage.values()))

def _phase_total_wall_ms(samples: list[TaskSample], phase: str, *, layer_ids: list[int] | None=None) -> float:
    layers = layer_ids
    if layers is None:
        layers = sorted({sample.layer_id for sample in samples if sample.phase == phase})
    return sum((_phase_layer_wall_ms(samples, phase, layer_id) for layer_id in layers))

def _timing_summary(samples: list[TaskSample], *, d: int, local_total_wall_ms: float) -> dict[str, float]:
    layer_ids = list(range(d))
    input_vss_ms = _phase_total_wall_ms(samples, 'INPUT_VSS', layer_ids=[-1])
    rg_prep_ms = _phase_total_wall_ms(samples, 'LAYER_RG_PREP', layer_ids=layer_ids)
    mul_mask_prep_ms = _phase_total_wall_ms(samples, 'LAYER_MUL_MASK_PREP', layer_ids=layer_ids)
    mul_online_ms = _phase_total_wall_ms(samples, 'LAYER_MUL_ONLINE', layer_ids=layer_ids)
    output_recon_ms = _phase_total_wall_ms(samples, 'OUTPUT_RECON', layer_ids=[d])
    critical_path_ms = input_vss_ms + rg_prep_ms + mul_mask_prep_ms + mul_online_ms + output_recon_ms
    total_cpu_ms = sum((sample.cpu_ms for sample in samples))
    return {'input_vss_ms': _ms(input_vss_ms), 'rg_prep_ms': _ms(rg_prep_ms), 'mul_mask_prep_ms': _ms(mul_mask_prep_ms), 'mul_online_ms': _ms(mul_online_ms), 'output_recon_ms': _ms(output_recon_ms), 'total_wall_ms': _ms(local_total_wall_ms), 'critical_path_ms': _ms(critical_path_ms), 'total_cpu_ms': _ms(total_cpu_ms)}

def _aggregate_batch_rows(*, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, success: bool, error: str, samples: list[TaskSample], rg_mode: str=DEFAULT_RG_MODE, machine_count: int=NUM_MACHINES) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[TaskSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.phase, sample.layer_id, sample.stage].append(sample)
    rows: list[dict[str, Any]] = []
    phase_layer_stage_order: list[tuple[str, int, str]] = []
    phase_layer_stage_order.extend((('INPUT_VSS', -1, stage) for stage in INPUT_VSS_STAGES))
    for layer_id in range(d):
        phase_layer_stage_order.extend((('LAYER_RG_PREP', layer_id, stage) for stage in RG_PREP_STAGES))
        phase_layer_stage_order.extend((('LAYER_MUL_MASK_PREP', layer_id, stage) for stage in MUL_MASK_STAGES))
        phase_layer_stage_order.extend((('LAYER_MUL_ONLINE', layer_id, stage) for stage in MUL_ONLINE_STAGES))
    phase_layer_stage_order.append(('OUTPUT_RECON', d, 'output_recon'))
    for phase, layer_id, stage in phase_layer_stage_order:
        batch_samples = grouped.get((phase, layer_id, stage), [])
        expected = _expected_task_count(phase=phase, layer_id=layer_id, stage=stage, num_input_secrets=num_input_secrets, d=d, w=w, n=n, rg_mode=rg_mode)
        task_ids = {sample.task_id for sample in batch_samples}
        task_count = len(task_ids) if task_ids else expected
        machine_cpu: list[list[float]] = [[] for _ in range(machine_count)]
        wall_values: list[float] = []
        for sample in batch_samples:
            machine_cpu[sample.rank % machine_count].append(sample.cpu_ms)
            wall_values.append(sample.wall_ms)
        row: dict[str, Any] = {'scheme': scheme, 'protocol': PROTOCOL, 'n': n, 't': t, 'd': d, 'w': w, 'num_input_secrets': num_input_secrets, 'repeat_id': repeat_id, 'phase': phase, 'layer_id': layer_id, 'stage': stage, 'task_count': task_count, 'wall_max_ms': _ms(max(wall_values) if wall_values else 0.0), 'success': 'true' if success else 'false', 'error': error}
        for machine_id in range(NUM_MACHINES):
            values = machine_cpu[machine_id] if machine_id < machine_count else []
            row[f'machine{machine_id}_cpu_sum_ms'] = _ms(sum(values))
            row[f'machine{machine_id}_cpu_max_ms'] = _ms(max(values) if values else 0.0)
        rows.append(row)
    return rows

def _build_stage_rank_outputs(*, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, success: bool, error: str, samples: list[TaskSample], comm: dict[str, Any], batch_rows: list[dict[str, Any]], rg_mode: str=DEFAULT_RG_MODE) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_task_counts = {(row['phase'], int(row['layer_id']), row['stage']): int(row['task_count']) for row in batch_rows}
    stage_comm = cube_stage_comm_map(n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, field_bytes=8, rg_mode=rg_mode, input_vss_stages=INPUT_VSS_STAGES, rg_prep_stages=RG_PREP_STAGES, mul_mask_stages=MUL_MASK_STAGES, mul_online_stages=MUL_ONLINE_STAGES, comm_row=comm)
    return build_stage_outputs(scheme=scheme, protocol=PROTOCOL, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, success=success, error=error, samples=samples, stage_comm=stage_comm, stage_task_counts=stage_task_counts)

def _rectangular_plain_values(depth: int, width: int, inputs: dict[int, int]) -> list[int]:
    prev = [mod(inputs[j]) for j in range(width)]
    for _layer_id in range(depth):
        prev = [mod(prev[j] * prev[(j + 1) % width]) for j in range(width)]
    return prev

def _input_values(num_input_secrets: int, *, seed: int) -> dict[int, int]:
    rng = Random(seed)
    return {idx: rand_field(rng) for idx in range(num_input_secrets)}

def _input_dealer(input_id: int, n: int) -> int:
    return input_id % n

def _clear_mul_scratch(node: Any, layer: int, mul_inst: str, random_keys: list[Any], t: int) -> None:
    node.clear_instance_state('mul', mul_inst)
    node.clear_instance_state('rs', f'{mul_inst}:reshare-r0')
    state = node.get_state(layer)
    for key in random_keys + resharing_prep_keys(mul_inst, t):
        state.values.pop(key, None)
    state.values.pop((mul_inst, 'r0_mask_share'), None)
    node.get_state(layer + 1).values.pop((mul_inst, 'r0_reshared'), None)

@dataclass(frozen=True)
class _GateCtx:
    gate_id: int
    left: WireRef
    right: WireRef
    target_layer: int
    start_layer: int
    mul_inst: str
    rg_inst: str

async def _run_vss_recorded(*, node: Any, recorder: list[TaskSample], detailed_writer: csv.DictWriter | None, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, phase: str, layer_id: int, task_id: int, call_type: str, gate_id: int, vss_round_id: int, start_layer: int, dealer_rank: int, secret: int, instance_id: str, rng: Random) -> int:
    start = time.perf_counter()
    share = await run_vss(node=node, start_layer=start_layer, dealer_rank=dealer_rank, secret=secret, instance_id=instance_id, rng=rng)
    local_wall_ms = (time.perf_counter() - start) * 1000.0
    stages = dict(_ps(node, start_layer).get(instance_id, {}).get('stage_metrics', {}))
    _record_vss_tasks(recorder, phase=phase, layer_id=layer_id, task_id=task_id, rank=node.rank, stages=stages, total_wall_ms=local_wall_ms)
    if detailed_writer is not None:
        row = _base_call_row(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, rank=node.rank, call_type=call_type, layer_id=layer_id, gate_id=gate_id, vss_round_id=vss_round_id)
        _copy_vss_stage_metrics(row, stages)
        row['local_call_wall_ms'] = _ms(local_wall_ms)
        detailed_writer.writerow(row)
    return share

async def _run_rg_for_gate(*, node: Any, recorder: list[TaskSample], detailed_writer: csv.DictWriter | None, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, start_layer: int, layer_id: int, gate_id: int, instance_id: str, rng: Random, parallel_limit: int) -> list[int]:
    round_rows: list[dict[str, Any]] = []

    async def _one_vss(dealer_rank: int) -> tuple[int, int, dict[str, float], float]:
        vss_id = f'{instance_id}:vss:{dealer_rank}'
        secret = rand_field(rng) if node.rank == dealer_rank else 0
        start = time.perf_counter()
        share_i = await run_vss(node=node, start_layer=start_layer, dealer_rank=dealer_rank, secret=secret, instance_id=vss_id, rng=rng)
        local_wall_ms = (time.perf_counter() - start) * 1000.0
        stages = dict(_ps(node, start_layer).get(vss_id, {}).get('stage_metrics', {}))
        return (dealer_rank, share_i, stages, local_wall_ms)
    vss_results = await bounded_gather([_one_vss(dealer_rank) for dealer_rank in range(n)], min(n, parallel_limit))
    shares_of_m: dict[int, int] = {}
    for dealer_rank, share_i, stages, local_wall_ms in vss_results:
        shares_of_m[dealer_rank + 1] = share_i
        task_id = gate_id * n + dealer_rank
        _record_vss_tasks(recorder, phase='LAYER_RG_PREP', layer_id=layer_id, task_id=task_id, rank=node.rank, stages=stages, total_wall_ms=local_wall_ms)
        if detailed_writer is not None:
            row = _base_call_row(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, rank=node.rank, call_type='RG_FOR_MUL', layer_id=layer_id, gate_id=gate_id, vss_round_id=dealer_rank)
            _copy_vss_stage_metrics(row, stages)
            row['local_call_wall_ms'] = _ms(local_wall_ms)
            round_rows.append(row)
        node.clear_instance_state('vss', f'{instance_id}:vss:{dealer_rank}')
    combine_cpu_start_ns = time.process_time_ns()
    combine_wall_start_ns = time.perf_counter_ns()
    output = combine_vandermonde_shares(shares_of_m, n, n - t)
    combine_cpu_ms = (time.process_time_ns() - combine_cpu_start_ns) / 1000000.0
    combine_wall_ms = (time.perf_counter_ns() - combine_wall_start_ns) / 1000000.0
    _record_task(recorder, phase='LAYER_RG_PREP', layer_id=layer_id, stage='rg_combine', task_id=gate_id, rank=node.rank, cpu_ms=combine_cpu_ms, wall_ms=combine_wall_ms)
    if detailed_writer is not None and round_rows:
        round_rows.sort(key=lambda row: int(row.get('vss_round_id', 0)))
        round_rows[-1]['rg_combine_cpu_ms'] = _ms(combine_cpu_ms)
        round_rows[-1]['rg_combine_wall_ms'] = _ms(combine_wall_ms)
        round_rows[-1]['local_call_wall_ms'] = _ms(float(round_rows[-1]['local_call_wall_ms']) + combine_wall_ms)
        detailed_writer.writerows(round_rows)
    return output

async def _run_rg_batch_for_layer(*, node: Any, recorder: list[TaskSample], detailed_writer: csv.DictWriter | None, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, start_layer: int, layer_id: int, instance_id: str, rng: Random) -> list[list[int]]:
    batch_size = w
    start = time.perf_counter()
    records = await run_rg_batch(node=node, start_layer=start_layer, batch_size=batch_size, instance_id=instance_id, layer_id=layer_id, purpose='mul_mask', rng=rng)
    local_wall_ms = (time.perf_counter() - start) * 1000.0
    for dealer_rank in range(n):
        vss_id = f'{instance_id}:vss:{dealer_rank}'
        stages = dict(_ps(node, start_layer).get(vss_id, {}).get('stage_metrics', {}))
        _record_vss_tasks(recorder, phase='LAYER_RG_PREP', layer_id=layer_id, task_id=dealer_rank, rank=node.rank, stages=stages, total_wall_ms=local_wall_ms / max(n, 1))
        if detailed_writer is not None:
            row = _base_call_row(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, rank=node.rank, call_type='RG_BATCH_FOR_LAYER', layer_id=layer_id, gate_id=-1, vss_round_id=dealer_rank)
            _copy_vss_stage_metrics(row, stages)
            row['local_call_wall_ms'] = _ms(local_wall_ms / max(n, 1))
            detailed_writer.writerow(row)
        node.clear_instance_state('vss', vss_id)
    combine_cpu_start_ns = time.process_time_ns()
    combine_wall_start_ns = time.perf_counter_ns()
    outputs = [record.values for record in records]
    combine_cpu_ms = (time.process_time_ns() - combine_cpu_start_ns) / 1000000.0
    combine_wall_ms = (time.perf_counter_ns() - combine_wall_start_ns) / 1000000.0
    per_gate_combine_cpu = combine_cpu_ms / max(batch_size, 1)
    per_gate_combine_wall = combine_wall_ms / max(batch_size, 1)
    for gate_id in range(batch_size):
        _record_task(recorder, phase='LAYER_RG_PREP', layer_id=layer_id, stage='rg_combine', task_id=gate_id, rank=node.rank, cpu_ms=per_gate_combine_cpu, wall_ms=per_gate_combine_wall)
    return outputs

def _populate_mul_rs_prep(node: Any, layer: int, instance_id: str, rng: Random) -> None:
    state = node.get_state(layer)
    keys = resharing_prep_keys(instance_id, node.mpc_config.t)
    for key in keys:
        secret = rand_field(rng)
        shares = shamir_share(secret, node.n, node.mpc_config.t, rng=rng)
        state.values[key] = shares[party_id(node.rank)]

async def _run_rank_async(*, rank: int, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, field_bytes: int, scheme: str, base_port: int, startup_delay: float, instance_id: str, input_values: dict[int, int], debug_detailed: bool, call_csv_path: str, samples_path: str, parallel_limit: int, rg_mode: str=DEFAULT_RG_MODE, layer_parallel: bool=False, transport_payload_mode: str=TRANSPORT_OBJECT, enable_vss_profiling: bool=True, hosts: tuple[str, ...] | None=None) -> RankRawResult:
    from cubempc.network.node import NodeProcess
    reset_counters()
    reset_cache_stats()
    gate_parallel = w if layer_parallel else parallel_limit
    vss_profiler = VssProfiler(enabled=enable_vss_profiling, rank_id=rank, phase='LAYER_RG_PREP', layer_id=-1, stage='')
    set_profiler(vss_profiler)
    _ = field_bytes
    mpc = MPCConfig.create(n=n, t=t, base_port=base_port, hosts=hosts)
    node = NodeProcess(rank, n, base_port, mpc)
    node.transport_payload_mode = transport_payload_mode
    await node.start_server()
    await asyncio.sleep(startup_delay)
    await node.connect_peers()
    local_start = time.perf_counter()
    rng = Random(_seed(instance_id, 'rank', rank))
    recorder: list[TaskSample] = []
    detailed_writer: csv.DictWriter | None = None
    detailed_fh = None
    if debug_detailed:
        detailed_fh = Path(call_csv_path).open('w', newline='', encoding='utf-8')
        detailed_writer = csv.DictWriter(detailed_fh, fieldnames=CALL_FIELDS)
        detailed_writer.writeheader()
    try:
        wires: list[WireRef] = []
        for input_id in range(num_input_secrets):
            dealer_rank = _input_dealer(input_id, n)
            vss_id = f'{instance_id}:input:{input_id}'
            secret = mod(input_values[input_id]) if rank == dealer_rank else 0
            await _run_vss_recorded(node=node, recorder=recorder, detailed_writer=detailed_writer, scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, phase='INPUT_VSS', layer_id=-1, task_id=input_id, call_type='INPUT_VSS', gate_id=input_id, vss_round_id=-1, start_layer=0, dealer_rank=dealer_rank, secret=secret, instance_id=vss_id, rng=rng)
            wires.append(WireRef(layer=3, key=_share_key(vss_id)))
        prev = [wires[j % num_input_secrets] for j in range(w)]
        for layer_id in range(d):
            vss_profiler.layer_id = layer_id
            gate_ctxs: list[_GateCtx] = []
            for gate_id in range(w):
                left = prev[gate_id]
                right = prev[(gate_id + 1) % w]
                if left.layer != right.layer:
                    raise RuntimeError('rectangular multiplication circuit should not need WIRE_RS')
                target_layer = left.layer
                mul_inst = f'{instance_id}:mul:L{layer_id}:G{gate_id}'
                gate_ctxs.append(_GateCtx(gate_id=gate_id, left=left, right=right, target_layer=target_layer, start_layer=target_layer - 3, mul_inst=mul_inst, rg_inst=f'{mul_inst}:rg'))
            rg_shares_list: list[list[int]]
            if rg_mode == RG_MODE_BATCH_LAYER:
                layer_rg_inst = f'{instance_id}:rg:L{layer_id}'
                rg_shares_list = await _run_rg_batch_for_layer(node=node, recorder=recorder, detailed_writer=detailed_writer, scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, start_layer=gate_ctxs[0].start_layer, layer_id=layer_id, instance_id=layer_rg_inst, rng=Random(_seed(layer_rg_inst, 'rng')))
            else:
                rg_shares_list = await bounded_gather([_run_rg_for_gate(node=node, recorder=recorder, detailed_writer=detailed_writer, scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, start_layer=ctx.start_layer, layer_id=layer_id, gate_id=ctx.gate_id, instance_id=ctx.rg_inst, rng=Random(_seed(ctx.rg_inst, 'rng')), parallel_limit=parallel_limit) for ctx in gate_ctxs], parallel_limit)

            async def _mask_prep(ctx: _GateCtx, rg_shares: list[int]) -> tuple[_GateCtx, list[Any], dict[str, float]]:
                random_keys = map_rg_output_to_mul_keys(node, ctx.target_layer, ctx.mul_inst, rg_shares, t)
                _populate_mul_rs_prep(node, ctx.target_layer, ctx.mul_inst, Random(_seed(ctx.mul_inst, 'rs-prep')))
                mul_timing: dict[str, float] = {}
                await run_mul(node, ctx.target_layer, ctx.left.key, ctx.right.key, random_keys, (instance_id, 'wire', layer_id, ctx.gate_id), ctx.mul_inst, recv_timeout=240.0, timing=mul_timing, phase='mask_prep')
                return (ctx, random_keys, mul_timing)
            mask_results = await bounded_gather([_mask_prep(ctx, rg_shares) for ctx, rg_shares in zip(gate_ctxs, rg_shares_list)], gate_parallel)
            for ctx, _random_keys, mul_timing in mask_results:
                for stage in MUL_MASK_STAGES:
                    _record_task(recorder, phase='LAYER_MUL_MASK_PREP', layer_id=layer_id, stage=stage, task_id=ctx.gate_id, rank=rank, cpu_ms=float(mul_timing.get(f'{stage}_cpu_ms', 0.0)), wall_ms=float(mul_timing.get(f'{stage}_wall_ms', 0.0)))
                if detailed_writer is not None:
                    mask_row = _base_call_row(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, rank=rank, call_type='MUL_MASK_PREP', layer_id=layer_id, gate_id=ctx.gate_id, vss_round_id=-1)
                    _copy_timing(mask_row, mul_timing, ['mul_mask_compute_cpu_ms', 'mul_mask_send_cpu_ms', 'mul_mask_compute_wall_ms', 'mul_mask_send_wall_ms'])
                    mask_row['local_call_wall_ms'] = _ms(mul_timing.get('mul_mask_compute_wall_ms', 0.0) + mul_timing.get('mul_mask_send_wall_ms', 0.0))
                    detailed_writer.writerow(mask_row)

            async def _mul_online(ctx: _GateCtx, random_keys: list[Any]) -> WireRef:
                output_key = (instance_id, 'wire', layer_id, ctx.gate_id)
                mul_timing: dict[str, float] = {}
                await run_mul(node, ctx.target_layer, ctx.left.key, ctx.right.key, random_keys, output_key, ctx.mul_inst, recv_timeout=240.0, timing=mul_timing, phase='online')
                for stage in MUL_ONLINE_STAGES:
                    _record_task(recorder, phase='LAYER_MUL_ONLINE', layer_id=layer_id, stage=stage, task_id=ctx.gate_id, rank=rank, cpu_ms=float(mul_timing.get(f'{stage}_cpu_ms', 0.0)), wall_ms=float(mul_timing.get(f'{stage}_wall_ms', 0.0)))
                if detailed_writer is not None:
                    online_row = _base_call_row(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, rank=rank, call_type='MUL_ONLINE', layer_id=layer_id, gate_id=ctx.gate_id, vss_round_id=-1)
                    _copy_timing(online_row, mul_timing, ['mul_mask_reconstruct_cpu_ms', 'mul_product_compute_cpu_ms', 'mul_product_broadcast_cpu_ms', 'mul_public_decode_cpu_ms', 'mul_rs_construct_cpu_ms', 'mul_rs_send_cpu_ms', 'mul_rs_decode_cpu_ms', 'mul_output_add_cpu_ms', 'mul_mask_reconstruct_wall_ms', 'mul_product_compute_wall_ms', 'mul_product_broadcast_wall_ms', 'mul_public_decode_wall_ms', 'mul_rs_construct_wall_ms', 'mul_rs_send_wall_ms', 'mul_rs_decode_wall_ms', 'mul_output_add_wall_ms'])
                    online_row['local_call_wall_ms'] = _ms(mul_timing.get('online_ms', 0.0))
                    detailed_writer.writerow(online_row)
                _clear_mul_scratch(node, ctx.target_layer, ctx.mul_inst, random_keys, t)
                return WireRef(layer=ctx.target_layer + 1, key=output_key)
            current = await bounded_gather([_mul_online(ctx, random_keys) for ctx, random_keys, _mul_timing in mask_results], gate_parallel)
            for wire in prev:
                node.get_state(wire.layer).values.pop(wire.key, None)
            if layer_id == 0:
                for input_id in range(num_input_secrets):
                    node.clear_instance_state('vss', f'{instance_id}:input:{input_id}')
            if layer_id >= 1:
                node.clear_before_layer(layer_id)
            prev = list(current)
    finally:
        if detailed_fh is not None:
            detailed_fh.close()
    local_total_wall_ms = (time.perf_counter() - local_start) * 1000.0
    await node.stop_server()
    with Path(samples_path).open('wb') as fh:
        pickle.dump(recorder, fh)
    counters = get_counters().to_dict()
    counters.update(get_cache_stats())
    vss_rows = [{'scheme': scheme, 'n': n, 't': t, 'd': d, 'w': w, 'repeat_id': repeat_id, 'phase': s.phase, 'layer_id': s.layer_id, 'stage': s.stage, 'rank_id': s.rank_id, 'op_name': s.op_name, 'op_count': s.op_count, 'cpu_ms': _ms(s.cpu_ms), 'wall_ms': _ms(s.wall_ms)} for s in vss_profiler.samples]
    return RankRawResult(rank=rank, output_share=mod(node.get_state(prev[0].layer).values[prev[0].key]), local_total_wall_ms=local_total_wall_ms, task_samples=recorder, call_csv_path=call_csv_path if debug_detailed else '', profile_counters=counters, vss_profile_samples=vss_rows)

def _worker_entry(*args: object) -> None:
    q: mp.Queue = args[-1]
    kwargs = args[0]
    try:
        result = asyncio.run(_run_rank_async(**kwargs))
        q.put(result)
    except Exception:
        q.put({'rank': kwargs.get('rank'), 'error': traceback.format_exc()})

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

def _next_free_base_port(n: int, *, candidate: int, used_ports: set[int]) -> int:
    base = candidate
    for _ in range(200):
        block = set(range(base, base + n))
        if base + n < 65535 and (not block & used_ports) and _port_block_available(base, n):
            used_ports.update(block)
            return base
        base += PORT_STRIDE
    raise RuntimeError(f'could not find a free port block of size {n}')

def _logical_comm(*, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, field_bytes: int, num_rs_calls: int, num_output_recon: int, rg_mode: str=DEFAULT_RG_MODE) -> dict[str, Any]:
    num_input_vss = num_input_secrets
    num_mul_gates = d * w
    num_mul_mask_prep_calls = num_mul_gates
    num_mul_online_calls = num_mul_gates
    if rg_mode == RG_MODE_BATCH_LAYER:
        num_rg_batch_calls = d
        num_rg_outputs = num_mul_gates
        num_rg_vss_instances = d * n
    else:
        num_rg_batch_calls = 0
        num_rg_outputs = num_mul_gates
        num_rg_vss_instances = num_mul_gates * n
    vss_dealer_p2p_bytes = n * (t + 1) ** 2 * field_bytes
    vss_com1_p2p_bytes = n * n * (t + 1) * field_bytes
    vss_com2_p2p_bytes = n * n * n * field_bytes
    vss_p2p_bytes = vss_dealer_p2p_bytes + vss_com1_p2p_bytes + vss_com2_p2p_bytes
    vss_broadcast_bytes = (n * n + 7) // 8
    input_vss_p2p_bytes = num_input_vss * vss_p2p_bytes
    input_vss_broadcast_bytes = num_input_vss * vss_broadcast_bytes
    rg_p2p_bytes = num_rg_vss_instances * w * vss_p2p_bytes if rg_mode == RG_MODE_BATCH_LAYER else num_rg_vss_instances * vss_p2p_bytes
    rg_broadcast_bytes = num_rg_vss_instances * w * vss_broadcast_bytes if rg_mode == RG_MODE_BATCH_LAYER else num_rg_vss_instances * vss_broadcast_bytes
    mul_mask_p2p_bytes = num_mul_mask_prep_calls * 2 * n * n * field_bytes
    mul_online_p2p_bytes = num_mul_online_calls * n * n * field_bytes
    mul_online_broadcast_bytes = num_mul_online_calls * n * field_bytes
    rs_p2p_bytes = num_rs_calls * n * n * field_bytes
    rs_broadcast_bytes = 0
    output_recon_p2p_bytes = num_output_recon * n * field_bytes
    output_recon_broadcast_bytes = 0
    total_p2p_bytes = input_vss_p2p_bytes + rg_p2p_bytes + mul_mask_p2p_bytes + mul_online_p2p_bytes + rs_p2p_bytes + output_recon_p2p_bytes
    total_broadcast_bytes = input_vss_broadcast_bytes + rg_broadcast_bytes + mul_online_broadcast_bytes + rs_broadcast_bytes + output_recon_broadcast_bytes
    vss_message_count = 2 * n * n + 2 * n
    if rg_mode == RG_MODE_BATCH_LAYER:
        rg_message_count = num_rg_batch_calls * n * vss_message_count
    else:
        rg_message_count = num_mul_gates * n * vss_message_count
    message_count = num_input_vss * vss_message_count + rg_message_count + num_mul_mask_prep_calls * n * n + num_mul_online_calls * (n + n * n) + num_rs_calls * n * n
    return {'scheme': scheme, 'protocol': PROTOCOL, 'n': n, 't': t, 'd': d, 'w': w, 'num_input_secrets': num_input_secrets, 'repeat_id': repeat_id, 'rg_mode': rg_mode, 'input_vss_p2p_bytes': input_vss_p2p_bytes, 'input_vss_broadcast_bytes': input_vss_broadcast_bytes, 'rg_p2p_bytes': rg_p2p_bytes, 'rg_broadcast_bytes': rg_broadcast_bytes, 'mul_mask_p2p_bytes': mul_mask_p2p_bytes, 'mul_online_p2p_bytes': mul_online_p2p_bytes, 'mul_online_broadcast_bytes': mul_online_broadcast_bytes, 'rs_p2p_bytes': rs_p2p_bytes, 'rs_broadcast_bytes': rs_broadcast_bytes, 'output_recon_p2p_bytes': output_recon_p2p_bytes, 'output_recon_broadcast_bytes': output_recon_broadcast_bytes, 'total_p2p_bytes': total_p2p_bytes, 'total_broadcast_bytes': total_broadcast_bytes, 'total_bytes': total_p2p_bytes + total_broadcast_bytes, 'message_count': message_count}

def _merge_profile_counters(results: list[RankRawResult]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for result in results:
        for key, value in result.profile_counters.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged

def _rg_counter_row(*, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, rg_mode: str, counters: dict[str, int]) -> dict[str, Any]:
    row: dict[str, Any] = {'scheme': scheme, 'n': n, 't': t, 'd': d, 'w': w, 'num_input_secrets': num_input_secrets, 'repeat_id': repeat_id, 'rg_mode': rg_mode}
    for field in RG_COUNTER_FIELDS:
        if field in row:
            continue
        row[field] = counters.get(field, 0)
    if rg_mode == RG_MODE_BATCH_LAYER:
        row['rg_vss_instance_count'] = row.get('rg_vss_instance_count', 0) or d * n
    else:
        row['rg_vss_instance_count'] = row.get('rg_vss_instance_count', 0) or d * w * n
    return row

def _count_row(*, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, success: bool, error: str, local_total_wall_ms: float, rg_mode: str=DEFAULT_RG_MODE, samples: list[TaskSample] | None=None) -> dict[str, Any]:
    num_mul_gates = d * w
    if rg_mode == RG_MODE_BATCH_LAYER:
        num_rg_calls = 0
        num_rg_batch_calls = d
    else:
        num_rg_calls = num_mul_gates
        num_rg_batch_calls = 0
    timing = _timing_summary(samples or [], d=d, local_total_wall_ms=local_total_wall_ms)
    return {'scheme': scheme, 'protocol': PROTOCOL, 'n': n, 't': t, 'd': d, 'w': w, 'num_input_secrets': num_input_secrets, 'repeat_id': repeat_id, 'rg_mode': rg_mode, 'success': 'true' if success else 'false', 'error': error, 'num_input_vss': num_input_secrets, 'num_mul_gates': num_mul_gates, 'num_rg_calls': num_rg_calls, 'num_rg_batch_calls': num_rg_batch_calls, 'num_rg_outputs': num_mul_gates, 'num_mul_mask_prep_calls': num_mul_gates, 'num_mul_online_calls': num_mul_gates, 'num_rs_calls': 0, 'num_output_recon': 0, **timing, 'local_total_wall_ms': _ms(local_total_wall_ms)}

def _load_task_samples(path: str) -> list[TaskSample]:
    with Path(path).open('rb') as fh:
        return pickle.load(fh)

def _append_rank_csvs(out_calls: Path, paths: list[str]) -> None:
    with out_calls.open('a', newline='', encoding='utf-8') as out_fh:
        for path in paths:
            if not path:
                continue
            with Path(path).open(newline='', encoding='utf-8') as in_fh:
                reader = csv.reader(in_fh)
                next(reader, None)
                writer = csv.writer(out_fh)
                writer.writerows(reader)

def _append_failure_call_rows(out_calls: Path, *, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, error: str) -> None:
    with out_calls.open('a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=CALL_FIELDS)
        for rank in range(n):
            row = _base_call_row(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, rank=rank, call_type='OUTPUT_RECON', layer_id=d, gate_id=-1, vss_round_id=-1, success=False, error=error)
            writer.writerow(row)

def _failure_batch_rows(*, scheme: str, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, error: str, rg_mode: str=DEFAULT_RG_MODE, machine_count: int=NUM_MACHINES) -> list[dict[str, Any]]:
    return _aggregate_batch_rows(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, success=False, error=error, samples=[], rg_mode=rg_mode, machine_count=machine_count)

def _rank_worker_kwargs(*, rank: int, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, field_bytes: int, scheme: str, base_port: int, startup_delay: float, instance_id: str, input_values: dict[int, int], debug_detailed: bool, call_csv_path: str, samples_path: str, parallel_limit: int, rg_mode: str, layer_parallel: bool, transport_payload_mode: str, hosts: tuple[str, ...] | None) -> dict[str, Any]:
    return {'rank': rank, 'n': n, 't': t, 'd': d, 'w': w, 'num_input_secrets': num_input_secrets, 'repeat_id': repeat_id, 'field_bytes': field_bytes, 'scheme': scheme, 'base_port': base_port, 'startup_delay': startup_delay, 'instance_id': instance_id, 'input_values': input_values, 'debug_detailed': debug_detailed, 'call_csv_path': call_csv_path, 'samples_path': samples_path, 'parallel_limit': parallel_limit, 'rg_mode': rg_mode, 'layer_parallel': layer_parallel, 'transport_payload_mode': transport_payload_mode, 'hosts': hosts}

def _run_one_repeat(*, n: int, t: int, d: int, w: int, num_input_secrets: int, repeat_id: int, field_bytes: int, scheme: str, base_port: int, startup_delay: float, timeout: float, debug_detailed: bool, out_calls: Path | None, tmp_dir: Path, parallel_limit: int, rg_mode: str=DEFAULT_RG_MODE, layer_parallel: bool=False, transport_payload_mode: str=TRANSPORT_OBJECT, host_layout: HostLayout | None=None, remote_repo: str | None=None, ssh_user: str | None=None, shared_tmp_dir: Path | None=None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    layout = host_layout or HostLayout.localhost()
    hosts = layout.hosts
    machine_count = layout.machine_count()
    instance_id = f'mpc-stage-raw-n{n}-d{d}-w{w}-r{repeat_id}'
    input_values = _input_values(num_input_secrets, seed=_seed(instance_id, 'inputs'))
    expected = _rectangular_plain_values(d, w, input_values)[0]
    run_start = time.perf_counter()
    results: list[RankRawResult] = []
    errors: list[str] = []
    if distributed_layout(layout) and shared_tmp_dir is None:
        raise ValueError('--shared-tmp-dir is required when --hosts spans multiple machines')
    work_dir = (shared_tmp_dir or tmp_dir) / instance_id
    work_dir.mkdir(parents=True, exist_ok=True)
    if distributed_layout(layout):
        if not remote_repo:
            raise ValueError('--remote-repo-dir is required when --hosts spans multiple machines')
        remote_procs: list[Any] = []
        for rank in range(n):
            call_csv_path = str(work_dir / f'{instance_id}-rank{rank}-calls.csv')
            samples_path = str(work_dir / f'{instance_id}-rank{rank}-samples.pkl')
            params_path = work_dir / f'rank{rank}-params.pkl'
            result_path = work_dir / f'rank{rank}-result.pkl'
            kwargs = _rank_worker_kwargs(rank=rank, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, field_bytes=field_bytes, scheme=scheme, base_port=base_port, startup_delay=startup_delay, instance_id=instance_id, input_values=input_values, debug_detailed=debug_detailed, call_csv_path=call_csv_path, samples_path=samples_path, parallel_limit=parallel_limit, rg_mode=rg_mode, layer_parallel=layer_parallel, transport_payload_mode=transport_payload_mode, hosts=hosts)
            with params_path.open('wb') as fh:
                pickle.dump(kwargs, fh)
            worker_args = ['--params-file', str(params_path), '--result-file', str(result_path)]
            remote_procs.append(launch_remote_rank(rank=rank, host=layout.host_for_rank(rank), remote_repo=remote_repo, python_executable=sys.executable, worker_args=worker_args, ssh_user=ssh_user))
        result_paths = [work_dir / f'rank{rank}-result.pkl' for rank in range(n)]
        wait_for_result_files(result_paths, timeout=timeout)
        for proc in remote_procs:
            rc = proc.wait(timeout=60)
            if rc != 0:
                errors.append(f'remote worker exit code {rc}')
        for rank, result_path in enumerate(result_paths):
            item = load_worker_result(result_path)
            if isinstance(item, RankRawResult):
                results.append(item)
            elif isinstance(item, dict) and 'error' in item:
                errors.append(f'rank {item.get('rank', rank)}: {item['error']}')
            else:
                errors.append(f'rank {rank}: invalid worker payload')
    else:
        ctx = mp.get_context('spawn')
        out_queue: mp.Queue = ctx.Queue()
        processes: list[mp.Process] = []
        for rank in range(n):
            call_csv_path = str(work_dir / f'{instance_id}-rank{rank}-calls.csv')
            samples_path = str(work_dir / f'{instance_id}-rank{rank}-samples.pkl')
            kwargs = _rank_worker_kwargs(rank=rank, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, field_bytes=field_bytes, scheme=scheme, base_port=base_port, startup_delay=startup_delay, instance_id=instance_id, input_values=input_values, debug_detailed=debug_detailed, call_csv_path=call_csv_path, samples_path=samples_path, parallel_limit=parallel_limit, rg_mode=rg_mode, layer_parallel=layer_parallel, transport_payload_mode=transport_payload_mode, hosts=hosts if not layout.is_local_only() else None)
            proc = ctx.Process(target=_worker_entry, args=(kwargs, out_queue), name=f'mpc-stage-raw-{rank}')
            proc.start()
            processes.append(proc)
        deadline = time.time() + timeout
        while len(results) + len(errors) < n and time.time() < deadline:
            try:
                item = out_queue.get(timeout=0.5)
            except Empty:
                continue
            if isinstance(item, RankRawResult):
                results.append(item)
            elif isinstance(item, dict) and 'error' in item:
                errors.append(f'rank {item.get('rank')}: {item['error']}')
        for proc in processes:
            proc.join(timeout=15)
            if proc.exitcode not in (0, None):
                errors.append(f'process {proc.name} exit code {proc.exitcode}')
    local_total_wall_ms = (time.perf_counter() - run_start) * 1000.0
    success = False
    error = ''
    if errors:
        error = ' | '.join(errors)
    elif len(results) != n:
        error = f'expected {n} rank results, got {len(results)}'
    else:
        results.sort(key=lambda r: r.rank)
        shares = {party_id(r.rank): r.output_share for r in results}
        reconstructed = reconstruct_output(shares, t)
        success = reconstructed == expected
        if not success:
            error = f'reconstructed {reconstructed} != expected {expected}'
        local_total_wall_ms = max((r.local_total_wall_ms for r in results))
    all_samples: list[TaskSample] = []
    if success:
        for result in sorted(results, key=lambda r: r.rank):
            samples_path = str(work_dir / f'{instance_id}-rank{result.rank}-samples.pkl')
            all_samples.extend(_load_task_samples(samples_path))
    if debug_detailed and out_calls is not None and results:
        _append_rank_csvs(out_calls, [r.call_csv_path for r in sorted(results, key=lambda r: r.rank)])
    if debug_detailed and out_calls is not None and (not success):
        _append_failure_call_rows(out_calls, scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, error=error)
    if success:
        batch_rows = _aggregate_batch_rows(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, success=True, error='', samples=all_samples, rg_mode=rg_mode, machine_count=machine_count)
    else:
        batch_rows = _failure_batch_rows(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, error=error, rg_mode=rg_mode, machine_count=machine_count)
    count = _count_row(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, success=success, error=error, local_total_wall_ms=local_total_wall_ms, rg_mode=rg_mode, samples=all_samples if success else None)
    comm = _logical_comm(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, field_bytes=field_bytes, num_rs_calls=0, num_output_recon=0, rg_mode=rg_mode)
    if comm['total_bytes'] != comm['total_p2p_bytes'] + comm['total_broadcast_bytes']:
        raise AssertionError('total_bytes sanity check failed')
    task_rows: list[dict[str, Any]] = []
    rank_agg_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rg_counter_row: dict[str, Any] = {}
    vss_profile_rows: list[dict[str, Any]] = []
    if success and all_samples:
        task_rows, rank_agg_rows, summary_rows = _build_stage_rank_outputs(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, success=True, error='', samples=all_samples, comm=comm, batch_rows=batch_rows, rg_mode=rg_mode)
        merged_counters = _merge_profile_counters(results)
        rg_counter_row = _rg_counter_row(scheme=scheme, n=n, t=t, d=d, w=w, num_input_secrets=num_input_secrets, repeat_id=repeat_id, rg_mode=rg_mode, counters=merged_counters)
        count['rg_vss_instance_count'] = rg_counter_row['rg_vss_instance_count']
        for result in results:
            vss_profile_rows.extend(result.vss_profile_samples)
    else:
        count['rg_vss_instance_count'] = d * n if rg_mode == RG_MODE_BATCH_LAYER else d * w * n
    return (batch_rows, count, comm, task_rows, rank_agg_rows, summary_rows, rg_counter_row, vss_profile_rows)

def _open_writer(path: Path, fields: list[str]) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open('w', newline='', encoding='utf-8')
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    fh.flush()
    return (fh, writer)

def _write_header(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

def _completed_count_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        return {(row['n'], row['d'], row['repeat_id']) for row in reader if row.get('success', '').lower() == 'true'}

def _prune_batch_rows_to_completed_counts(*, batch_path: Path, completed_keys: set[tuple[str, str, str]]) -> None:
    if not batch_path.exists():
        return
    tmp_path = batch_path.with_suffix(f'{batch_path.suffix}.tmp')
    with batch_path.open(newline='', encoding='utf-8') as in_fh, tmp_path.open('w', newline='', encoding='utf-8') as out_fh:
        reader = csv.DictReader(in_fh)
        writer = csv.DictWriter(out_fh, fieldnames=BATCH_FIELDS)
        writer.writeheader()
        for row in reader:
            key = (row.get('n', ''), row.get('d', ''), row.get('repeat_id', ''))
            if key in completed_keys:
                writer.writerow({field: row.get(field, '') for field in BATCH_FIELDS})
    tmp_path.replace(batch_path)

def main(argv: list[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description='CUBE end-to-end MPC batch-stage benchmark')
    parser.add_argument('--n', type=_parse_int_list, default=_parse_int_list('5'))
    parser.add_argument('--depth', '--d', type=_parse_int_list, default=_parse_int_list('2'), dest='depth')
    parser.add_argument('--width', '--w', type=int, default=10, dest='width')
    parser.add_argument('--num-input-secrets', type=int, default=None)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--field-bytes', type=int, default=8)
    parser.add_argument('--scheme', choices=[DEFAULT_SCHEME], default=DEFAULT_SCHEME)
    parser.add_argument('--rg-mode', choices=[RG_MODE_PER_GATE, RG_MODE_BATCH_LAYER], default=DEFAULT_RG_MODE, help='RG scheduling: per_gate (legacy) or batch_layer (one batched RG per layer)')
    parser.add_argument('--out-dir', type=Path, default=None, help='output directory; writes mpc_batch_stage_raw.csv, mpc_count_raw.csv, mpc_logical_comm_raw.csv, stage_rank_agg.csv, stage_summary.csv')
    parser.add_argument('--write-task-raw', action='store_true', help='also write compressed task-level stage_task_raw.csv.gz (debug only)')
    parser.add_argument('--debug-detailed', type=_parse_bool, default=False, help='also emit per-call/per-rank detailed CSV via --out-calls')
    parser.add_argument('--out-batch', type=Path, default=Path('bench_output/mpc_raw/mpc_batch_stage_raw.csv'))
    parser.add_argument('--out-calls', type=Path, default=None, help='detailed per-call CSV; only used when --debug-detailed true')
    parser.add_argument('--out-counts', type=Path, default=Path('bench_output/mpc_raw/mpc_count_raw.csv'))
    parser.add_argument('--out-stage-rank-agg', type=Path, default=Path('bench_output/mpc_raw/stage_rank_agg.csv'))
    parser.add_argument('--out-stage-task-raw', type=Path, default=Path('bench_output/mpc_raw/stage_task_raw.csv.gz'))
    parser.add_argument('--out-stage-summary', type=Path, default=Path('bench_output/mpc_raw/stage_summary.csv'))
    parser.add_argument('--out-comm', type=Path, default=Path('bench_output/mpc_raw/mpc_logical_comm_raw.csv'))
    parser.add_argument('--base-port', type=int, default=20000)
    parser.add_argument('--startup-delay', type=float, default=2.5)
    parser.add_argument('--timeout', type=float, default=7200.0)
    parser.add_argument('--cooldown', type=float, default=0.0)
    parser.add_argument('--parallel-limit', type=int, default=32, help='max concurrent gates / VSS instances per rank')
    parser.add_argument('--layer-parallel', type=_parse_bool, default=False, help='when true, run all w gates in a layer concurrently (parallel_limit=w)')
    parser.add_argument('--transport-payload-mode', choices=[TRANSPORT_JSON, TRANSPORT_OBJECT], default=TRANSPORT_OBJECT, help='object=pickle payloads (benchmark default); json=legacy JSON encoding')
    parser.add_argument('--resume', action='store_true', help='append missing successful repeats, pruning incomplete batch rows first')
    parser.add_argument('--hosts', type=str, default=None, help='comma-separated hostnames/IPs for 6-machine MPC (round-robin rank placement); default localhost')
    parser.add_argument('--remote-repo-dir', type=str, default=None, help='path to repo on each remote host (required for multi-host --hosts)')
    parser.add_argument('--ssh-user', type=str, default=None, help='optional SSH username for remote rank workers')
    parser.add_argument('--shared-tmp-dir', type=Path, default=None, help='shared directory visible at the same path on all hosts for worker params/results')
    args = parser.parse_args(argv)
    host_layout = HostLayout.parse(args.hosts)
    if args.out_dir is not None:
        args.out_batch = args.out_dir / 'mpc_batch_stage_raw.csv'
        args.out_counts = args.out_dir / 'mpc_count_raw.csv'
        args.out_comm = args.out_dir / 'mpc_logical_comm_raw.csv'
        args.out_stage_rank_agg = args.out_dir / 'stage_rank_agg.csv'
        args.out_stage_task_raw = args.out_dir / 'stage_task_raw.csv.gz'
        args.out_stage_summary = args.out_dir / 'stage_summary.csv'
        args.out_rg_counters = args.out_dir / 'cube_rg_internal_counters.csv'
        args.out_vss_profile = args.out_dir / 'cube_vss_internal_profile.csv'
    if args.repeat < 1:
        raise ValueError('--repeat must be >= 1')
    if args.width < 1:
        raise ValueError('--width must be >= 1')
    if args.debug_detailed and args.out_calls is None:
        raise ValueError('--out-calls is required when --debug-detailed true')
    num_input_secrets = args.num_input_secrets or args.width
    if num_input_secrets < args.width:
        raise ValueError('num_input_secrets must be at least width for this rectangular circuit')
    if args.parallel_limit < 1:
        raise ValueError('--parallel-limit must be >= 1')
    completed_keys = _completed_count_keys(args.out_counts) if args.resume else set()
    if args.resume:
        _prune_batch_rows_to_completed_counts(batch_path=args.out_batch, completed_keys=completed_keys)
        args.out_batch.parent.mkdir(parents=True, exist_ok=True)
        args.out_counts.parent.mkdir(parents=True, exist_ok=True)
        args.out_comm.parent.mkdir(parents=True, exist_ok=True)
        if not args.out_batch.exists():
            _write_header(args.out_batch, BATCH_FIELDS)
        if not args.out_counts.exists():
            _write_header(args.out_counts, COUNT_FIELDS)
        if not args.out_comm.exists():
            _write_header(args.out_comm, COMM_FIELDS)
        if not args.out_stage_rank_agg.exists():
            _write_header(args.out_stage_rank_agg, STAGE_RANK_AGG_FIELDS)
        if args.write_task_raw and (not args.out_stage_task_raw.exists()):
            append_gz_csv_rows(args.out_stage_task_raw, STAGE_TASK_RAW_FIELDS, [])
        if not args.out_stage_summary.exists():
            _write_header(args.out_stage_summary, STAGE_SUMMARY_FIELDS)
        batch_fh = args.out_batch.open('a', newline='', encoding='utf-8')
        batch_writer = csv.DictWriter(batch_fh, fieldnames=BATCH_FIELDS)
        count_fh = args.out_counts.open('a', newline='', encoding='utf-8')
        counts_writer = csv.DictWriter(count_fh, fieldnames=COUNT_FIELDS)
        comm_fh = args.out_comm.open('a', newline='', encoding='utf-8')
        comm_writer = csv.DictWriter(comm_fh, fieldnames=COMM_FIELDS)
        rank_agg_fh = args.out_stage_rank_agg.open('a', newline='', encoding='utf-8')
        rank_agg_writer = csv.DictWriter(rank_agg_fh, fieldnames=STAGE_RANK_AGG_FIELDS)
        summary_fh = args.out_stage_summary.open('a', newline='', encoding='utf-8')
        summary_writer = csv.DictWriter(summary_fh, fieldnames=STAGE_SUMMARY_FIELDS)
        rg_counter_path = getattr(args, 'out_rg_counters', Path('bench_output/mpc_raw/cube_rg_internal_counters.csv'))
        if not rg_counter_path.exists():
            _write_header(rg_counter_path, RG_COUNTER_FIELDS)
        rg_counter_fh = rg_counter_path.open('a', newline='', encoding='utf-8')
        rg_counter_writer = csv.DictWriter(rg_counter_fh, fieldnames=RG_COUNTER_FIELDS)
        vss_profile_path = getattr(args, 'out_vss_profile', Path('bench_output/mpc_raw/cube_vss_internal_profile.csv'))
        if not vss_profile_path.exists():
            _write_header(vss_profile_path, VSS_PROFILE_FIELDS)
        vss_profile_fh = vss_profile_path.open('a', newline='', encoding='utf-8')
        vss_profile_writer = csv.DictWriter(vss_profile_fh, fieldnames=VSS_PROFILE_FIELDS)
    else:
        batch_fh, batch_writer = _open_writer(args.out_batch, BATCH_FIELDS)
        count_fh, counts_writer = _open_writer(args.out_counts, COUNT_FIELDS)
        comm_fh, comm_writer = _open_writer(args.out_comm, COMM_FIELDS)
        rank_agg_fh, rank_agg_writer = _open_writer(args.out_stage_rank_agg, STAGE_RANK_AGG_FIELDS)
        summary_fh, summary_writer = _open_writer(args.out_stage_summary, STAGE_SUMMARY_FIELDS)
        rg_counter_path = getattr(args, 'out_rg_counters', Path('bench_output/mpc_raw/cube_rg_internal_counters.csv'))
        rg_counter_fh, rg_counter_writer = _open_writer(rg_counter_path, RG_COUNTER_FIELDS)
        vss_profile_path = getattr(args, 'out_vss_profile', Path('bench_output/mpc_raw/cube_vss_internal_profile.csv'))
        vss_profile_fh, vss_profile_writer = _open_writer(vss_profile_path, VSS_PROFILE_FIELDS)
    if not args.resume and args.debug_detailed and (args.out_calls is not None):
        _write_header(args.out_calls, CALL_FIELDS)
    used_ports: set[int] = set()
    port_cursor = 0
    try:
        with tempfile.TemporaryDirectory(prefix='mpc-stage-raw-') as tmp:
            tmp_dir = Path(tmp)
            for n in args.n:
                t = validate_party_count(n, None)
                for d in args.depth:
                    if d < 1:
                        raise ValueError(f'depth must be >= 1, got {d}')
                    for repeat_id in range(args.repeat):
                        key = (str(n), str(d), str(repeat_id))
                        if key in completed_keys:
                            print(f'MPC batch n={n} t={t} d={d} w={args.width} num_input_secrets={num_input_secrets} repeat={repeat_id}: skipped (resume)')
                            continue
                        base_port = _next_free_base_port(n, candidate=args.base_port + port_cursor, used_ports=used_ports)
                        port_cursor += PORT_STRIDE
                        task_rows: list[dict[str, Any]] = []
                        rank_agg_rows: list[dict[str, Any]] = []
                        summary_rows: list[dict[str, Any]] = []
                        rg_counter_row: dict[str, Any] = {}
                        vss_profile_rows: list[dict[str, Any]] = []
                        try:
                            batch_rows, count, comm, task_rows, rank_agg_rows, summary_rows, rg_counter_row, vss_profile_rows = _run_one_repeat(n=n, t=t, d=d, w=args.width, num_input_secrets=num_input_secrets, repeat_id=repeat_id, field_bytes=args.field_bytes, scheme=args.scheme, base_port=base_port, startup_delay=args.startup_delay, timeout=args.timeout, debug_detailed=args.debug_detailed, out_calls=args.out_calls, tmp_dir=tmp_dir, parallel_limit=args.parallel_limit, rg_mode=args.rg_mode, layer_parallel=args.layer_parallel, transport_payload_mode=args.transport_payload_mode, host_layout=host_layout, remote_repo=args.remote_repo_dir, ssh_user=args.ssh_user, shared_tmp_dir=args.shared_tmp_dir)
                        except Exception as exc:
                            error = f'{type(exc).__name__}: {exc}'
                            batch_rows = _failure_batch_rows(scheme=args.scheme, n=n, t=t, d=d, w=args.width, num_input_secrets=num_input_secrets, repeat_id=repeat_id, error=error, rg_mode=args.rg_mode)
                            if args.debug_detailed and args.out_calls is not None:
                                _append_failure_call_rows(args.out_calls, scheme=args.scheme, n=n, t=t, d=d, w=args.width, num_input_secrets=num_input_secrets, repeat_id=repeat_id, error=error)
                            count = _count_row(scheme=args.scheme, n=n, t=t, d=d, w=args.width, num_input_secrets=num_input_secrets, repeat_id=repeat_id, success=False, error=error, local_total_wall_ms=0.0, rg_mode=args.rg_mode)
                            comm = _logical_comm(scheme=args.scheme, n=n, t=t, d=d, w=args.width, num_input_secrets=num_input_secrets, repeat_id=repeat_id, field_bytes=args.field_bytes, num_rs_calls=0, num_output_recon=0, rg_mode=args.rg_mode)
                        batch_writer.writerows(batch_rows)
                        counts_writer.writerow(count)
                        comm_writer.writerow(comm)
                        if rank_agg_rows:
                            rank_agg_writer.writerows(rank_agg_rows)
                        if summary_rows:
                            summary_writer.writerows(summary_rows)
                        if rg_counter_row:
                            rg_counter_writer.writerow(rg_counter_row)
                        if vss_profile_rows:
                            vss_profile_writer.writerows(vss_profile_rows)
                        if args.write_task_raw and task_rows:
                            append_gz_csv_rows(args.out_stage_task_raw, STAGE_TASK_RAW_FIELDS, task_rows)
                        if count['success'] == 'true':
                            completed_keys.add(key)
                        batch_fh.flush()
                        count_fh.flush()
                        comm_fh.flush()
                        rank_agg_fh.flush()
                        summary_fh.flush()
                        rg_counter_fh.flush()
                        vss_profile_fh.flush()
                        print(f'MPC batch n={n} t={t} d={d} w={args.width} num_input_secrets={num_input_secrets} repeat={repeat_id}: {('ok' if count['success'] == 'true' else 'failed')} batch_rows={len(batch_rows)}')
                        print(f'sanity num_input_vss={count['num_input_vss']} num_mul_gates={count['num_mul_gates']} num_rg_calls={count['num_rg_calls']} num_rg_batch_calls={count['num_rg_batch_calls']} num_rg_outputs={count['num_rg_outputs']} num_mul_mask_prep_calls={count['num_mul_mask_prep_calls']} num_mul_online_calls={count['num_mul_online_calls']} total_bytes={comm['total_bytes']} p2p+broadcast={comm['total_p2p_bytes'] + comm['total_broadcast_bytes']}')
                        if args.cooldown > 0:
                            time.sleep(args.cooldown)
    finally:
        batch_fh.close()
        count_fh.close()
        comm_fh.close()
        rank_agg_fh.close()
        summary_fh.close()
        rg_counter_fh.close()
        vss_profile_fh.close()
    print(f'wrote {args.out_batch}')
    print(f'wrote {args.out_counts}')
    print(f'wrote {args.out_comm}')
    print(f'wrote {args.out_stage_rank_agg}')
    if args.write_task_raw:
        print(f'wrote {args.out_stage_task_raw}')
    print(f'wrote {args.out_stage_summary}')
    print(f'wrote {rg_counter_path}')
    print(f'wrote {vss_profile_path}')
    if args.debug_detailed and args.out_calls is not None:
        print(f'wrote {args.out_calls}')
if __name__ == '__main__':
    main()