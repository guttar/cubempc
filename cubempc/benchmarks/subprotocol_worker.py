from __future__ import annotations

import argparse
import asyncio
import importlib
import pickle
import traceback
from pathlib import Path
from typing import Any


def _load_params(path: Path) -> dict[str, Any]:
    with path.open('rb') as fh:
        return pickle.load(fh)


def _write_result(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as fh:
        pickle.dump(payload, fh)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Remote standalone sub-protocol rank worker')
    parser.add_argument('--params-file', type=Path, required=True)
    parser.add_argument('--result-file', type=Path, required=True)
    args = parser.parse_args(argv)
    params = _load_params(args.params_file)
    rank = params.get('kwargs', {}).get('rank', params.get('rank'))
    try:
        module = importlib.import_module(str(params['module']))
        func = getattr(module, str(params['function']))
        result = asyncio.run(func(**params['kwargs']))
        _write_result(args.result_file, result)
    except Exception:
        _write_result(args.result_file, {'rank': rank, 'error': traceback.format_exc()})


if __name__ == '__main__':
    main()
