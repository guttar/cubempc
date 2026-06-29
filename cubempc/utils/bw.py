from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class BWResult:
    poly: list[int]
    error_positions: list[int]


def mod_inv(a: int, p: int) -> int:
    a %= p
    if a == 0:
        raise ZeroDivisionError("inverse of zero")
    return pow(a, p - 2, p)


def poly_trim(poly: Sequence[int], p: int) -> list[int]:
    a = [x % p for x in poly]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    if not a:
        return [0]
    return a


def poly_eval(poly: Sequence[int], x: int, p: int) -> int:
    x %= p
    acc = 0
    for c in reversed(poly):
        acc = (acc * x + c) % p
    return acc


def poly_divmod(num: Sequence[int], den: Sequence[int], p: int) -> tuple[list[int], list[int]]:
    n = poly_trim(num, p)
    d = poly_trim(den, p)

    if len(d) == 1 and d[0] == 0:
        raise ZeroDivisionError("polynomial division by zero")

    if len(n) < len(d):
        return [0], n

    q = [0] * (len(n) - len(d) + 1)
    r = n[:]
    inv_lc = mod_inv(d[-1], p)

    while len(r) >= len(d) and not (len(r) == 1 and r[0] == 0):
        shift = len(r) - len(d)
        coeff = r[-1] * inv_lc % p
        q[shift] = coeff

        for i in range(len(d)):
            r[shift + i] = (r[shift + i] - coeff * d[i]) % p

        r = poly_trim(r, p)

    return poly_trim(q, p), poly_trim(r, p)


def solve_linear_mod(A: list[list[int]], b: list[int], p: int) -> Optional[list[int]]:
    if not A:
        return []

    m = len(A)
    n = len(A[0])

    M = []
    for row, rhs in zip(A, b):
        if len(row) != n:
            raise ValueError("inconsistent row length")
        M.append([v % p for v in row] + [rhs % p])

    rank = 0
    pivots: list[int] = []

    for col in range(n):
        pivot = None
        for r in range(rank, m):
            if M[r][col] % p != 0:
                pivot = r
                break

        if pivot is None:
            continue

        M[rank], M[pivot] = M[pivot], M[rank]

        inv = mod_inv(M[rank][col], p)
        for c in range(col, n + 1):
            M[rank][c] = M[rank][c] * inv % p

        for r in range(m):
            if r == rank:
                continue
            factor = M[r][col] % p
            if factor != 0:
                for c in range(col, n + 1):
                    M[r][c] = (M[r][c] - factor * M[rank][c]) % p

        pivots.append(col)
        rank += 1

        if rank == m:
            break

    for r in range(m):
        if all(M[r][c] % p == 0 for c in range(n)) and M[r][n] % p != 0:
            return None

    sol = [0] * n
    for row_idx, col in enumerate(pivots):
        sol[col] = M[row_idx][n] % p

    return sol


def berlekamp_welch_decode(
    xs: Sequence[int],
    ys: Sequence[int],
    degree: int,
    max_errors: int,
    p: int,
) -> Optional[BWResult]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have same length")

    m = len(xs)
    d = degree
    e = max_errors

    if d < 0 or e < 0:
        raise ValueError("degree and max_errors must be non-negative")

    if m < d + 2 * e + 1:
        raise ValueError(f"not enough points for BW: got {m}, need at least {d + 2 * e + 1}")

    xs_mod = [x % p for x in xs]
    ys_mod = [y % p for y in ys]

    if len(set(xs_mod)) != len(xs_mod):
        raise ValueError("evaluation points must be distinct modulo p")

    num_q = d + e + 1
    num_a = e

    A: list[list[int]] = []
    b: list[int] = []

    for x, y in zip(xs_mod, ys_mod):
        powers = [1]
        for _ in range(1, max(d + e, e) + 1):
            powers.append(powers[-1] * x % p)

        row = []

        for k in range(num_q):
            row.append(powers[k])

        for k in range(num_a):
            row.append((-y * powers[k]) % p)

        A.append(row)
        b.append(y * powers[e] % p)

    sol = solve_linear_mod(A, b, p)
    if sol is None:
        return None

    Q = poly_trim(sol[:num_q], p)
    E = [1] if e == 0 else poly_trim(sol[num_q:] + [1], p)

    P, rem = poly_divmod(Q, E, p)

    if not (len(rem) == 1 and rem[0] == 0):
        return None

    P = poly_trim(P, p)

    if len(P) - 1 > d:
        return None

    error_positions = []
    for idx, (x, y) in enumerate(zip(xs_mod, ys_mod)):
        if poly_eval(P, x, p) != y:
            error_positions.append(idx)

    if len(error_positions) > e:
        return None

    if len(P) < d + 1:
        P = P + [0] * (d + 1 - len(P))

    return BWResult(poly=P, error_positions=error_positions)


def berlekamp_welch_decode_vector(
    xs: Sequence[int],
    y_vectors: Sequence[Sequence[int]],
    degree: int,
    max_errors: int,
    p: int,
) -> Optional[tuple[list[list[int]], list[int]]]:
    if len(xs) != len(y_vectors):
        raise ValueError("xs and y_vectors must have same length")

    if not y_vectors:
        raise ValueError("empty y_vectors")

    width = len(y_vectors[0])
    if width == 0:
        raise ValueError("empty coefficient vector")

    for v in y_vectors:
        if len(v) != width:
            raise ValueError("all vectors must have the same length")

    component_polys: list[list[int]] = []

    for c in range(width):
        ys_c = [vec[c] % p for vec in y_vectors]
        res = berlekamp_welch_decode(xs, ys_c, degree, max_errors, p)
        if res is None:
            return None
        component_polys.append(res.poly)

    error_positions: list[int] = []
    for idx, x in enumerate(xs):
        recovered_vec = [poly_eval(component_polys[c], x, p) for c in range(width)]
        original_vec = [v % p for v in y_vectors[idx]]
        if recovered_vec != original_vec:
            error_positions.append(idx)

    if len(error_positions) > max_errors:
        return None

    return component_polys, error_positions
