from __future__ import annotations
from random import Random
from cubempc.field import div, mod, mul, neg, sub
from cubempc.poly import UniPoly
from cubempc.profiling import bump as profile_bump
from cubempc.rs_decode import berlekamp_welch_decode

def shamir_share(secret: int, n: int, t: int, rng: Random | None=None) -> dict[int, int]:
    if n < 1:
        raise ValueError(f'n must be >= 1, got {n}')
    if t < 0:
        raise ValueError(f't must be >= 0, got {t}')
    f = UniPoly.random(t, constant=secret, rng=rng)
    return {i: f.eval(i) for i in range(1, n + 1)}

def shamir_reconstruct(shares: dict[int, int], t: int) -> int:
    if t < 0:
        raise ValueError(f't must be >= 0, got {t}')
    if len(shares) < t + 1:
        raise ValueError(f'need at least {t + 1} shares, got {len(shares)}')
    ranks = sorted(shares.keys())[:t + 1]
    points = [(r, shares[r]) for r in ranks]
    profile_bump(lagrange_eval_count=1, fast_decode_count=1)
    return UniPoly.interpolate(points).coeff(0)

def robust_reconstruct(shares: dict[int, int], degree: int, max_errors: int) -> int | None:
    if not shares:
        return None
    if max_errors == 0:
        profile_bump(fast_path_count=1, fast_decode_count=1)
        try:
            return shamir_reconstruct(shares, degree)
        except ValueError:
            return None
    points = [(rank, value) for rank, value in shares.items()]
    profile_bump(bw_decode_count=1, fallback_decode_count=1)
    poly = berlekamp_welch_decode(points, degree, max_errors)
    if poly is None:
        return None
    return poly.coeff(0)

def lagrange_coeffs_at_zero(xs: list[int]) -> list[int]:
    if len(xs) < 1:
        raise ValueError('xs must be non-empty')
    xs_mod = [mod(x) for x in xs]
    m = len(xs_mod)
    lambdas: list[int] = []
    for i in range(m):
        xi = xs_mod[i]
        num, den = (1, 1)
        for j in range(m):
            if i == j:
                continue
            xj = xs_mod[j]
            num = mul(num, neg(xj))
            den = mul(den, sub(xi, xj))
        lambdas.append(div(num, den))
    return lambdas