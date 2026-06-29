from __future__ import annotations

import argparse
import asyncio
import pickle
import sys
import traceback
from pathlib import Path

from cubempc.benchmarks.run_mpc_stage_raw import RankRawResult, _run_rank_async


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run one MPC stage-raw rank worker')
    parser.add_argument('--params-file', type=Path, required=True)
    parser.add_argument('--result-file', type=Path, required=True)
    args = parser.parse_args(argv)
    with args.params_file.open('rb') as fh:
        kwargs = pickle.load(fh)
    try:
        result = asyncio.run(_run_rank_async(**kwargs))
        payload: RankRawResult | dict[str, object] = result
    except Exception:
        payload = {'rank': kwargs.get('rank'), 'error': traceback.format_exc()}
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    with args.result_file.open('wb') as fh:
        pickle.dump(payload, fh)


if __name__ == '__main__':
    main(sys.argv[1:])
