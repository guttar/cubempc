from __future__ import annotations
import io
import math
from dataclasses import dataclass
FIELD_ELEMENT_BYTES = 8
SUBPROTOCOL_CSV_FIELDS = ['protocol', 'n', 't', 'repeat_id', 'total_time_ms', 'logical_p2p_bytes', 'logical_broadcast_bytes', 'logical_total_bytes', 'logical_message_count', 'success']
MPC_CSV_FIELDS = ['protocol', 'n', 't', 'depth', 'width', 'mul_ratio', 'repeat_id', 'total_time_ms', 'logical_p2p_bytes', 'logical_broadcast_bytes', 'logical_total_bytes', 'logical_message_count', 'success']

def _vss_logical(n: int, t: int) -> tuple[int, int, int]:
    dealer_to_com1 = n * (t + 1) * (t + 1)
    com1_to_com2 = n * n * (t + 1)
    com2_to_com3 = n * n * n
    p2p_field_elements = dealer_to_com1 + com1_to_com2 + com2_to_com3
    p2p_bytes = p2p_field_elements * FIELD_ELEMENT_BYTES
    broadcast_bits = n * n
    broadcast_bytes = math.ceil(broadcast_bits / 8)
    message_count = n + n * n + n + n * n
    return (p2p_bytes, broadcast_bytes, message_count)

@dataclass
class LogicalMetrics:
    logical_p2p_bytes: int = 0
    logical_broadcast_bytes: int = 0
    logical_message_count: int = 0

    @property
    def logical_total_bytes(self) -> int:
        return self.logical_p2p_bytes + self.logical_broadcast_bytes

    def add_vss_cost(self, n: int, t: int) -> None:
        p2p, bcast, mc = _vss_logical(n, t)
        self.logical_p2p_bytes += p2p
        self.logical_broadcast_bytes += bcast
        self.logical_message_count += mc

    def add_rg_cost(self, n: int, t: int) -> None:
        p2p, bcast, mc = _vss_logical(n, t)
        self.logical_p2p_bytes += n * p2p
        self.logical_broadcast_bytes += n * bcast
        self.logical_message_count += n * mc

    def add_rs_cost(self, n: int, t: int, *, include_preprocessing: bool=False) -> None:
        self.logical_p2p_bytes += n * n * FIELD_ELEMENT_BYTES
        self.logical_message_count += n * n
        if include_preprocessing:
            self.add_rg_cost(n, t)

    def add_mul_cost(self, n: int, t: int, *, include_preprocessing: bool=False) -> None:
        mask_eval_field_elements = 2 * n * n
        rs_field_elements = n * n
        self.logical_p2p_bytes += (mask_eval_field_elements + rs_field_elements) * FIELD_ELEMENT_BYTES
        self.logical_broadcast_bytes += n * FIELD_ELEMENT_BYTES
        self.logical_message_count += n * n + n + n * n
        if include_preprocessing:
            self.add_rg_cost(n, t)

    def merge(self, other: LogicalMetrics) -> None:
        self.logical_p2p_bytes += other.logical_p2p_bytes
        self.logical_broadcast_bytes += other.logical_broadcast_bytes
        self.logical_message_count += other.logical_message_count

    def to_row(self) -> dict[str, int]:
        return {'logical_p2p_bytes': self.logical_p2p_bytes, 'logical_broadcast_bytes': self.logical_broadcast_bytes, 'logical_total_bytes': self.logical_total_bytes, 'logical_message_count': self.logical_message_count}

def vss_logical_cost(n: int, t: int) -> dict[str, int]:
    lm = LogicalMetrics()
    lm.add_vss_cost(n, t)
    return lm.to_row()

def vss_stage_logical_cost(n: int, t: int) -> dict[str, dict[str, int]]:
    dealer_fe = n * (t + 1) * (t + 1)
    com1_fe = n * n * (t + 1)
    com2_scalar_fe = n * n * n
    broadcast_bits = n * n
    broadcast_bytes = math.ceil(broadcast_bits / 8)
    dealer_p2p = dealer_fe * FIELD_ELEMENT_BYTES
    com1_p2p = com1_fe * FIELD_ELEMENT_BYTES
    com2_scalar_p2p = com2_scalar_fe * FIELD_ELEMENT_BYTES
    total_p2p = dealer_p2p + com1_p2p + com2_scalar_p2p
    return {'Dealer': {'p2p_bytes': dealer_p2p, 'broadcast_bytes': 0, 'total_bytes': dealer_p2p, 'message_count': n}, 'Com1': {'p2p_bytes': com1_p2p, 'broadcast_bytes': 0, 'total_bytes': com1_p2p, 'message_count': n * n}, 'Com2Decode': {'p2p_bytes': 0, 'broadcast_bytes': 0, 'total_bytes': 0, 'message_count': 0}, 'Com2Broadcast': {'p2p_bytes': 0, 'broadcast_bytes': broadcast_bytes, 'total_bytes': broadcast_bytes, 'message_count': n}, 'Com2SendScalar': {'p2p_bytes': com2_scalar_p2p, 'broadcast_bytes': 0, 'total_bytes': com2_scalar_p2p, 'message_count': n * n}, 'Public': {'p2p_bytes': 0, 'broadcast_bytes': 0, 'total_bytes': 0, 'message_count': 0}, 'Com3': {'p2p_bytes': 0, 'broadcast_bytes': 0, 'total_bytes': 0, 'message_count': 0}, 'Total': {'p2p_bytes': total_p2p, 'broadcast_bytes': broadcast_bytes, 'total_bytes': total_p2p + broadcast_bytes, 'message_count': n + n * n + n + n * n}}

def rg_logical_cost(n: int, t: int) -> dict[str, int]:
    lm = LogicalMetrics()
    lm.add_rg_cost(n, t)
    return lm.to_row()

def rs_logical_cost(n: int, t: int, *, include_preprocessing: bool=False) -> dict[str, int]:
    lm = LogicalMetrics()
    lm.add_rs_cost(n, t, include_preprocessing=include_preprocessing)
    return lm.to_row()

def mul_logical_cost(n: int, t: int, *, include_preprocessing: bool=False) -> dict[str, int]:
    lm = LogicalMetrics()
    lm.add_mul_cost(n, t, include_preprocessing=include_preprocessing)
    return lm.to_row()

def mpc_logical_cost(circuit: object, n: int, t: int, *, randomness_mode: str='local') -> dict[str, int]:
    if randomness_mode not in {'local', 'rg'}:
        raise ValueError(f'unsupported randomness_mode {randomness_mode!r}')
    lm = LogicalMetrics()
    wires: dict[str, int] = {}
    rg_available_by_layer: dict[int, int] = {}

    def consume_random(layer: int, count: int) -> None:
        if randomness_mode == 'local' or count <= 0:
            return
        q = n - t
        available = rg_available_by_layer.get(layer, 0)
        while available < count:
            lm.add_rg_cost(n, t)
            available += q
        rg_available_by_layer[layer] = available - count
    inputs = list(getattr(circuit, 'inputs'))
    for wire in inputs:
        lm.add_vss_cost(n, t)
        wires[wire] = 3
    for gate in getattr(circuit, 'gates'):
        op = getattr(gate, 'op')
        gid = getattr(gate, 'gid')
        if op == 'input':
            continue
        in1 = getattr(gate, 'in1')
        in2 = getattr(gate, 'in2')
        if op == 'add':
            target = max(wires[in1], wires[in2])
            for wire in {in1, in2}:
                while wires[wire] < target:
                    consume_random(wires[wire], t)
                    lm.add_rs_cost(n, t)
                    wires[wire] += 1
            wires[gid] = target
        elif op == 'cmul':
            wires[gid] = wires[in1]
        elif op == 'mul':
            target = max(wires[in1], wires[in2])
            for wire in {in1, in2}:
                while wires[wire] < target:
                    consume_random(wires[wire], t)
                    lm.add_rs_cost(n, t)
                    wires[wire] += 1
            consume_random(target, 3 * t + 1)
            consume_random(target, t)
            lm.add_mul_cost(n, t)
            wires[gid] = target + 1
        else:
            raise ValueError(f'unsupported gate op {op!r}')
    return lm.to_row()

@dataclass
class Metrics:
    serialized_p2p_bytes: int = 0
    serialized_broadcast_bytes: int = 0
    message_count: int = 0
    physical_bytes: int = 0

    def total_serialized_bytes(self) -> int:
        return self.serialized_p2p_bytes + self.serialized_broadcast_bytes

    def record_p2p(self, size: int | None=None) -> None:
        if size is not None:
            if size < 0:
                raise ValueError('negative metric increment')
            self.serialized_p2p_bytes += size
            self.physical_bytes += size
        self.message_count += 1

    def record_broadcast(self, size: int, *, physical_size: int | None=None) -> None:
        if size < 0:
            raise ValueError('negative metric increment')
        self.serialized_broadcast_bytes += size
        self.message_count += 1
        if physical_size is not None:
            if physical_size < 0:
                raise ValueError('negative physical_size')
            self.physical_bytes += physical_size
        else:
            self.physical_bytes += size

    def merge(self, other: Metrics) -> None:
        self.serialized_p2p_bytes += other.serialized_p2p_bytes
        self.serialized_broadcast_bytes += other.serialized_broadcast_bytes
        self.message_count += other.message_count
        self.physical_bytes += other.physical_bytes

    def to_row(self, **extra: str | int | float) -> dict[str, str | int | float]:
        row: dict[str, str | int | float] = {'serialized_p2p_bytes': self.serialized_p2p_bytes, 'serialized_broadcast_bytes': self.serialized_broadcast_bytes, 'serialized_total_bytes': self.total_serialized_bytes(), 'message_count': self.message_count, 'physical_bytes': self.physical_bytes}
        row.update(extra)
        return row
CommMetrics = Metrics
CSV_HEADER = ['experiment', 'protocol', 'n', 't', 'num_layers', 'logical_p2p_bytes', 'logical_broadcast_bytes', 'logical_total_bytes', 'logical_message_count', 'wall_ms']

def write_csv(path: str, rows: list[dict[str, str | int | float]]) -> None:
    from cubempc.csv_io import write_aligned_csv
    write_aligned_csv(path, CSV_HEADER, rows)

def format_csv(rows: list[dict[str, str | int | float]]) -> str:
    from cubempc.csv_io import COLUMN_SEP, _column_widths, _format_row
    widths = _column_widths(CSV_HEADER, rows)
    header = COLUMN_SEP.join((field.ljust(widths[field]) for field in CSV_HEADER))
    body = '\n'.join((_format_row(CSV_HEADER, widths, row) for row in rows))
    return header + '\n' + body + '\n'