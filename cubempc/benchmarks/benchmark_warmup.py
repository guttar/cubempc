from __future__ import annotations
import logging
import time
from collections.abc import Callable
from typing import Any
logger = logging.getLogger(__name__)

def run_external_warmup(runner: Callable[..., Any], *, warmup: bool, pause: float, **run_kwargs: Any) -> None:
    if not warmup:
        return
    try:
        runner(**run_kwargs)
        logger.info('warmup ok')
    except Exception as exc:
        logger.warning('warmup failed: %s', exc)
    if pause > 0:
        time.sleep(pause)