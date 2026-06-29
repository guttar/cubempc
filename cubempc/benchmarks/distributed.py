from __future__ import annotations

import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cubempc.host_layout import HostLayout


def ssh_target(host: str, ssh_user: str | None) -> str:
    if ssh_user:
        return f'{ssh_user}@{host}'
    return host


def launch_remote_rank(
    *,
    rank: int,
    host: str,
    remote_repo: str,
    python_executable: str,
    worker_args: list[str],
    ssh_user: str | None,
    worker_module: str = 'cubempc.benchmarks.mpc_stage_worker',
) -> subprocess.Popen[bytes]:
    quoted_args = ' '.join(shlex.quote(arg) for arg in worker_args)
    remote_cmd = f'cd {shlex.quote(remote_repo)} && {shlex.quote(python_executable)} -m {shlex.quote(worker_module)} {quoted_args}'
    cmd = ['ssh', ssh_target(host, ssh_user), remote_cmd]
    return subprocess.Popen(cmd)


def wait_for_result_files(
    result_paths: list[Path],
    *,
    timeout: float,
    poll_interval: float = 0.5,
) -> None:
    deadline = time.time() + timeout
    pending = set(result_paths)
    while pending and time.time() < deadline:
        for path in list(pending):
            if path.exists() and path.stat().st_size > 0:
                pending.discard(path)
        if pending:
            time.sleep(poll_interval)
    if pending:
        missing = ', '.join(str(path) for path in sorted(pending))
        raise TimeoutError(f'timed out waiting for remote worker results: {missing}')


def load_worker_result(path: Path) -> Any:
    with path.open('rb') as fh:
        return pickle.load(fh)


def distributed_layout(hosts: HostLayout) -> bool:
    return not hosts.is_local_only()
