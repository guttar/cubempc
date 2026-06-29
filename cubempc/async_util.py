from __future__ import annotations
import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar
T = TypeVar('T')

async def bounded_gather(coros: Iterable[Awaitable[T]], limit: int) -> list[T]:
    if limit < 1:
        raise ValueError(f'limit must be >= 1, got {limit}')
    tasks = list(coros)
    if not tasks:
        return []
    sem = asyncio.Semaphore(limit)

    async def _run(coro: Awaitable[T]) -> T:
        async with sem:
            return await coro
    return await asyncio.gather(*(_run(c) for c in tasks))