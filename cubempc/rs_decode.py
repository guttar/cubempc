from __future__ import annotations
from cubempc.field import P, add, inv, mod, mul, neg, sub
from cubempc.poly import BiPoly, UniPoly, _x_pow
from cubempc.utils.bw import berlekamp_welch_decode as _strict_bw_decode
from cubempc.utils.bw import berlekamp_welch_decode_vector

def solve_linear_system_mod_p(A: list[list[int]], b: list[int]) -> list[int] | None:
    m = len(A)
    if m == 0:
        return [] if len(b) == 0 else None
    n = len(A[0])
    if len(b) != m:
        raise ValueError('A row count must match b length')
    for row in A:
        if len(row) != n:
            raise ValueError('A must be rectangular')
    aug = [[mod(A[i][j]) for j in range(n)] + [mod(b[i])] for i in range(m)]
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(n):
        sel = None
        for r in range(pivot_row, m):
            if aug[r][col] != 0:
                sel = r
                break
        if sel is None:
            continue
        aug[pivot_row], aug[sel] = (aug[sel], aug[pivot_row])
        piv = aug[pivot_row][col]
        inv_piv = inv(piv)
        for c in range(col, n + 1):
            aug[pivot_row][c] = mul(aug[pivot_row][c], inv_piv)
        for r in range(m):
            if r == pivot_row:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for c in range(col, n + 1):
                aug[r][c] = sub(aug[r][c], mul(factor, aug[pivot_row][c]))
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row >= m:
            break
    for r in range(pivot_row, m):
        if aug[r][n] != 0 and all((aug[r][c] == 0 for c in range(n))):
            return None
    if len(pivot_cols) < n:
        return None
    sol = [0] * n
    for ri in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[ri]
        val = aug[ri][n]
        for j in range(pc + 1, n):
            val = sub(val, mul(aug[ri][j], sol[j]))
        sol[pc] = val
    for i in range(m):
        check = 0
        for j in range(n):
            check = add(check, mul(A[i][j], sol[j]))
        if mod(check) != mod(b[i]):
            return None
    return sol

def _count_matches(p: UniPoly, points: list[tuple[int, int]]) -> int:
    return sum((1 for x, y in points if p.eval(x) == mod(y)))

def _divide_monic(numer: UniPoly, denom: UniPoly) -> UniPoly | None:
    de = denom.degree()
    if de < 0:
        return None
    if denom.coeff(de) != 1:
        return None
    n = [numer.coeff(i) for i in range(numer.degree() + 1)]
    d = [denom.coeff(i) for i in range(de + 1)]
    while len(n) > 1 and n[-1] == 0:
        n.pop()
    if not n:
        n = [0]
    if len(n) - 1 < de:
        if len(n) == 1 and n[0] == 0:
            return UniPoly([0])
        return None
    q_len = len(n) - de
    q = [0] * q_len
    for _ in range(q_len):
        while len(n) > 1 and n[-1] == 0:
            n.pop()
        if len(n) <= de:
            break
        shift = len(n) - 1 - de
        lead = n[-1]
        if shift < 0 or shift >= len(q):
            return None
        q[shift] = lead
        for i in range(de + 1):
            idx = i + shift
            if idx < len(n):
                n[idx] = sub(n[idx], mul(lead, d[i]))
    while len(n) > 1 and n[-1] == 0:
        n.pop()
    if len(n) == 1 and n[0] == 0:
        return UniPoly(q)
    return None

def _berlekamp_welch_decode_slow(pts: list[tuple[int, int]], degree: int, max_errors: int) -> UniPoly | None:
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    try:
        result = _strict_bw_decode(xs, ys, degree, max_errors, P)
    except ValueError:
        return None
    if result is None:
        return None
    return UniPoly(result.poly)

def berlekamp_welch_decode(points: list[tuple[int, int]], degree: int, max_errors: int, *, fast_path: bool=True) -> UniPoly | None:
    if degree < 0 or max_errors < 0:
        raise ValueError('degree and max_errors must be non-negative')
    n_pts = len(points)
    if n_pts == 0:
        return UniPoly([0]) if degree == 0 else None
    pts = [(mod(x), mod(y)) for x, y in points]
    min_agree = n_pts - max_errors
    if fast_path and n_pts >= degree + 1:
        sample = pts[:degree + 1]
        xs = [x for x, _ in sample]
        if len(set(xs)) == len(sample):
            candidate = UniPoly.interpolate(sample)
            if candidate.degree() <= degree and _count_matches(candidate, pts) >= min_agree:
                return candidate
    return _berlekamp_welch_decode_slow(pts, degree, max_errors)

def decode_poly_valued(points: list[tuple[int, UniPoly]], degree: int, max_errors: int, value_degree: int) -> BiPoly | None:
    if value_degree < 0:
        raise ValueError('value_degree must be non-negative')
    xs = [x for x, _ in points]
    y_vectors = [[y.coeff(k) for k in range(value_degree + 1)] for _, y in points]
    try:
        decoded = berlekamp_welch_decode_vector(xs, y_vectors, degree, max_errors, P)
    except ValueError:
        return None
    if decoded is None:
        return None
    component_polys, _errors = decoded
    rows: list[list[int]] = []
    for a in range(degree + 1):
        rows.append([component_polys[b][a] for b in range(value_degree + 1)])
    return BiPoly(rows)

def decode_polynomial(points: dict[int, int], degree: int, max_errors: int=0) -> UniPoly | None:
    pts = list(points.items())
    return berlekamp_welch_decode(pts, degree, max_errors)

def decode_poly_valued_batch(lines_by_batch: list[dict[int, UniPoly]], degree: int, max_errors: int, value_degree: int, *, fast_path: bool=True) -> list[BiPoly | None]:
    from cubempc.profiling import bump as profile_bump
    batch_size = len(lines_by_batch)
    if batch_size == 0:
        return []
    profile_bump(vss_decode_count=batch_size, decode_fast_batch_count=1)
    out: list[BiPoly | None] = []
    for lines in lines_by_batch:
        xs = sorted(lines.keys())
        if len(xs) < degree + 2 * max_errors + 1:
            out.append(None)
            continue
        if fast_path and max_errors == 0:
            profile_bump(fast_decode_count=1)
        else:
            profile_bump(decode_scalar_count=value_degree + 1, bw_decode_count=value_degree + 1)
        out.append(decode_poly_valued([(x, lines[x]) for x in xs], degree, max_errors, value_degree))
    return out