from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class FieldVector:
    values: list[int]

    @property
    def batch_size(self) -> int:
        return len(self.values)

@dataclass(slots=True)
class FieldMatrix:
    rows: list[list[int]]

    @property
    def nrows(self) -> int:
        return len(self.rows)

    @property
    def ncols(self) -> int:
        return len(self.rows[0]) if self.rows else 0

def scalar_table_payload(*, j_party: int, k_party: int, batch_size: int, table: list[list[int]]) -> dict:
    return {'kind': 'scalar_table', 'j_party': j_party, 'k_party': k_party, 'batch_size': batch_size, 'table': table}

def parse_scalar_table(payload: dict) -> tuple[int, int, int, list[list[int]]]:
    if payload.get('kind') == 'scalar_table':
        return (int(payload['j_party']), int(payload['k_party']), int(payload['batch_size']), payload['table'])
    j_party = int(payload['j_party'])
    k_party = int(payload['k_party'])
    vals_list = payload['vals_list']
    batch_size = len(vals_list)
    n = max((max((int(k) for k in vals)) for vals in vals_list if vals), default=0)
    table = [[0] * batch_size for _ in range(n)]
    for batch_idx, vals in enumerate(vals_list):
        for i_party, val in vals.items():
            table[int(i_party) - 1][batch_idx] = int(val)
    return (j_party, k_party, batch_size, table)