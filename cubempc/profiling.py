from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ProtocolCounters:
    rg_call_count: int = 0
    rg_batch_count: int = 0
    rg_output_count: int = 0
    rg_vss_instance_count: int = 0
    rg_vss_dealer_count: int = 0
    rg_vss_secret_count: int = 0
    vss_poly_sample_count: int = 0
    vss_encode_count: int = 0
    vss_decode_count: int = 0
    vss_encode_count: int = 0
    vss_broadcast_msg_count: int = 0
    vss_send_scalar_count: int = 0
    vss_reconstruct_count: int = 0
    lagrange_eval_count: int = 0
    bw_decode_count: int = 0
    fast_decode_count: int = 0
    fallback_decode_count: int = 0
    fast_path_count: int = 0
    correction_triggered_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    com3_vector_reconstruct_count: int = 0
    com3_scalar_reconstruct_count: int = 0
    decode_fast_batch_count: int = 0
    decode_scalar_count: int = 0
    lagrange_cache_hit: int = 0
    lagrange_cache_miss: int = 0
    broadcast_payload_count: int = 0
    broadcast_scalar_count: int = 0

    def merge(self, other: ProtocolCounters) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def bump(self, **kwargs: int) -> None:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(f'unknown counter {key!r}')
            setattr(self, key, getattr(self, key) + value)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}
_global_counters = ProtocolCounters()

def reset_counters() -> None:
    global _global_counters
    _global_counters = ProtocolCounters()

def get_counters() -> ProtocolCounters:
    return _global_counters

def bump(**kwargs: int) -> None:
    _global_counters.bump(**kwargs)