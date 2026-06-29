from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence
COLUMN_SEP = ',  '

def format_cell(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        text = f'{value:.6f}'.rstrip('0').rstrip('.')
        return text or '0'
    return str(value)

def _column_widths(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    widths: dict[str, int] = {}
    for field in fieldnames:
        cells = [field] + [format_cell(row.get(field, '')) for row in rows]
        widths[field] = max((len(cell) for cell in cells))
    return widths

def _format_row(fieldnames: Sequence[str], widths: dict[str, int], row: Mapping[str, Any]) -> str:
    parts = [format_cell(row.get(field, '')).ljust(widths[field]) for field in fieldnames]
    return COLUMN_SEP.join(parts)

def write_aligned_csv(path: str | Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    widths = _column_widths(fieldnames, rows)
    header_parts = [field.ljust(widths[field]) for field in fieldnames]
    lines = [COLUMN_SEP.join(header_parts)]
    lines.extend((_format_row(fieldnames, widths, row) for row in rows))
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def read_aligned_csv(path: str | Path, expected_fieldnames: Sequence[str] | None=None) -> list[dict[str, str]]:
    out = Path(path)
    if not out.exists():
        return []
    with out.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        fieldnames = [name.strip() for name in reader.fieldnames or []]
        if expected_fieldnames is not None and fieldnames != list(expected_fieldnames):
            return []
        rows: list[dict[str, str]] = []
        for raw in reader:
            rows.append({key.strip(): value.strip() if value is not None else '' for key, value in raw.items()})
        return rows