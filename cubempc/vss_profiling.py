from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

@dataclass
class OpSample:
    phase: str = ''
    layer_id: int = -1
    stage: str = ''
    rank_id: int = -1
    op_name: str = ''
    op_count: int = 1
    cpu_ms: float = 0.0
    wall_ms: float = 0.0

@dataclass
class VssProfiler:
    samples: list[OpSample] = field(default_factory=list)
    phase: str = ''
    layer_id: int = -1
    stage: str = ''
    rank_id: int = -1
    enabled: bool = True

    def reset(self) -> None:
        self.samples.clear()

    @contextmanager
    def op(self, name: str, *, count: int=1) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        cpu0 = time.process_time_ns()
        wall0 = time.perf_counter_ns()
        yield
        cpu_ms = (time.process_time_ns() - cpu0) / 1000000.0
        wall_ms = (time.perf_counter_ns() - wall0) / 1000000.0
        self.samples.append(OpSample(phase=self.phase, layer_id=self.layer_id, stage=self.stage, rank_id=self.rank_id, op_name=name, op_count=count, cpu_ms=cpu_ms, wall_ms=wall_ms))

    def merge(self, other: VssProfiler) -> None:
        self.samples.extend(other.samples)
_global_profiler = VssProfiler(enabled=False)

def get_profiler() -> VssProfiler:
    return _global_profiler

def set_profiler(profiler: VssProfiler) -> None:
    global _global_profiler
    _global_profiler = profiler

def enable_profiling(*, rank_id: int, phase: str, layer_id: int, stage: str) -> VssProfiler:
    profiler = VssProfiler(enabled=True, rank_id=rank_id, phase=phase, layer_id=layer_id, stage=stage)
    set_profiler(profiler)
    return profiler

def disable_profiling() -> list[OpSample]:
    global _global_profiler
    samples = list(_global_profiler.samples)
    _global_profiler = VssProfiler(enabled=False)
    return samples