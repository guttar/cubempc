from __future__ import annotations
from functools import lru_cache
from typing import Any
from cubempc.field import mod
from cubempc.shamir import lagrange_coeffs_at_zero
cache_hit_count: int = 0
cache_miss_count: int = 0

def reset_cache_stats() -> None:
    global cache_hit_count, cache_miss_count
    cache_hit_count = 0
    cache_miss_count = 0

def get_cache_stats() -> dict[str, int]:
    return {'cache_hit_count': cache_hit_count, 'cache_miss_count': cache_miss_count}

def _cached_call(fn, *args: int):
    global cache_hit_count, cache_miss_count
    misses_before = fn.cache_info().misses
    result = fn(*args)
    if fn.cache_info().misses > misses_before:
        cache_miss_count += 1
    else:
        cache_hit_count += 1
    return result

@lru_cache(maxsize=64)
def _evaluation_points(n: int) -> tuple[int, ...]:
    return tuple(range(1, n + 1))

@lru_cache(maxsize=64)
def _lagrange_coefficients(n: int, t: int) -> tuple[int, ...]:
    xs = list(range(1, t + 2))
    return tuple(lagrange_coeffs_at_zero(xs))

@lru_cache(maxsize=64)
def _vandermonde_matrix(n: int, q: int) -> tuple[tuple[int, ...], ...]:
    from cubempc.field import pow_mod
    return tuple((tuple((pow_mod(i + 1, ell) for ell in range(q))) for i in range(n)))

@lru_cache(maxsize=64)
def _degree_reduction_weights(n: int, t: int) -> tuple[int, ...]:
    return tuple(lagrange_coeffs_at_zero(list(range(1, n + 1))[:t + 1]))

def evaluation_points(n: int) -> list[int]:
    return list(_cached_call(_evaluation_points, n))

def lagrange_coefficients(n: int, t: int) -> list[int]:
    return list(_cached_call(_lagrange_coefficients, n, t))

def reconstruction_coefficients(n: int, t: int) -> list[int]:
    return lagrange_coefficients(n, t)

def vandermonde_matrix(n: int, q: int) -> list[list[int]]:
    rows = _cached_call(_vandermonde_matrix, n, q)
    return [list(row) for row in rows]

def vandermonde_inverse(n: int, q: int) -> list[list[int]] | None:
    return None

def degree_reduction_weights(n: int, t: int) -> list[int]:
    return list(_cached_call(_degree_reduction_weights, n, t))

@lru_cache(maxsize=128)
def _lagrange_coefficients_for_points(xs_key: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(lagrange_coeffs_at_zero(list(xs_key)))

def lagrange_coefficients_for_points(xs: list[int]) -> list[int]:
    key = tuple((mod(x) for x in xs))
    return list(_cached_call(_lagrange_coefficients_for_points, key))

def snapshot_cache_info() -> dict[str, Any]:
    stats = get_cache_stats()
    return {**stats, 'evaluation_points_entries': _evaluation_points.cache_info().currsize, 'lagrange_entries': _lagrange_coefficients.cache_info().currsize, 'vandermonde_entries': _vandermonde_matrix.cache_info().currsize}