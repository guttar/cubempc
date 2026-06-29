from __future__ import annotations
from random import Random
from cubempc.field import add, div, mod, mul, rand_field, sub

def _strip_uni(coeffs: list[int]) -> list[int]:
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
    return coeffs

def _zero_pad_row(row: list[int], length: int) -> list[int]:
    if len(row) >= length:
        return row[:length]
    return row + [0] * (length - len(row))

def _x_pow(x: int, exp: int) -> int:
    r = 1
    for _ in range(exp):
        r = mul(r, x)
    return r

class UniPoly:
    __slots__ = ('coeffs',)

    def __init__(self, coeffs: list[int]) -> None:
        self.coeffs = _strip_uni([mod(c) for c in coeffs])

    def degree(self) -> int:
        return len(self.coeffs) - 1

    def coeff(self, d: int) -> int:
        if d < 0:
            return 0
        if d >= len(self.coeffs):
            return 0
        return self.coeffs[d]

    def eval(self, x: int) -> int:
        x = mod(x)
        result = 0
        power = 1
        for c in self.coeffs:
            result = add(result, mul(c, power))
            power = mul(power, x)
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UniPoly):
            return NotImplemented
        return self.coeffs == other.coeffs

    def __add__(self, other: UniPoly) -> UniPoly:
        n = max(len(self.coeffs), len(other.coeffs))
        return UniPoly([add(self.coeff(d), other.coeff(d)) for d in range(n)])

    def __mul__(self, other: UniPoly | int) -> UniPoly:
        if isinstance(other, int):
            scalar = mod(other)
            if scalar == 0:
                return UniPoly([0])
            return UniPoly([mul(c, scalar) for c in self.coeffs])
        out = [0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, ai in enumerate(self.coeffs):
            for j, bj in enumerate(other.coeffs):
                out[i + j] = add(out[i + j], mul(ai, bj))
        return UniPoly(out)
    __rmul__ = __mul__

    @classmethod
    def random(cls, deg: int, constant: int | None=None, rng: Random | None=None) -> UniPoly:
        if deg < 0:
            raise ValueError(f'degree must be non-negative, got {deg}')
        coeffs = [rand_field(rng) for _ in range(deg + 1)]
        if constant is not None:
            coeffs[0] = mod(constant)
        return cls(coeffs)

    @classmethod
    def interpolate(cls, points: list[tuple[int, int]]) -> UniPoly:
        m = len(points)
        if m == 0:
            return cls([0])
        poly = cls([0])
        for i in range(m):
            xi, yi = points[i]
            xi = mod(xi)
            yi = mod(yi)
            basis = cls([1])
            denom = 1
            for j in range(m):
                if i == j:
                    continue
                xj, _ = points[j]
                xj = mod(xj)
                basis = basis * cls([sub(0, xj), 1])
                denom = mul(denom, sub(xi, xj))
            poly = poly + basis * div(yi, denom)
        return poly

def _strip_bivariate(rows: list[list[int]]) -> list[list[int]]:
    if not rows:
        return [[0]]
    rows = [[mod(c) for c in row] for row in rows]
    while len(rows) > 1 and all((c == 0 for c in rows[-1])):
        rows.pop()
    max_cols = max((len(r) for r in rows))
    rows = [_zero_pad_row(r, max_cols) for r in rows]
    while max_cols > 1 and all((rows[a][max_cols - 1] == 0 for a in range(len(rows)))):
        max_cols -= 1
        rows = [r[:max_cols] for r in rows]
    return rows if rows else [[0]]

class BiPoly:
    __slots__ = ('coeffs',)

    def __init__(self, coeffs: list[list[int]]) -> None:
        self.coeffs = _strip_bivariate(coeffs)

    def deg_x(self) -> int:
        return len(self.coeffs) - 1

    def deg_y(self) -> int:
        return max((len(row) for row in self.coeffs)) - 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BiPoly):
            return NotImplemented
        return self.coeffs == other.coeffs

    def eval(self, x: int, y: int) -> int:
        x, y = (mod(x), mod(y))
        acc = 0
        for a, row in enumerate(self.coeffs):
            xpow = _x_pow(x, a)
            ypow = 1
            for b, c in enumerate(row):
                if c:
                    acc = add(acc, mul(c, mul(xpow, ypow)))
                ypow = mul(ypow, y)
        return acc

    def fix_x(self, x: int) -> UniPoly:
        x = mod(x)
        uni = [0] * (self.deg_y() + 1)
        for a, row in enumerate(self.coeffs):
            xpow = _x_pow(x, a)
            for b, c in enumerate(row):
                if c:
                    uni[b] = add(uni[b], mul(c, xpow))
        return UniPoly(uni)

    def fix_y(self, y: int) -> UniPoly:
        y = mod(y)
        uni = [0] * (self.deg_x() + 1)
        ypows = [1]
        for _ in range(self.deg_y()):
            ypows.append(mul(ypows[-1], y))
        for a, row in enumerate(self.coeffs):
            for b, c in enumerate(row):
                if c:
                    uni[a] = add(uni[a], mul(c, ypows[b]))
        return UniPoly(uni)

    @classmethod
    def random(cls, deg_x: int, deg_y: int, constant: int | None=None, rng: Random | None=None) -> BiPoly:
        if deg_x < 0 or deg_y < 0:
            raise ValueError('degrees must be non-negative')
        rows = [[rand_field(rng) for _ in range(deg_y + 1)] for _ in range(deg_x + 1)]
        if constant is not None:
            rows[0][0] = mod(constant)
        return cls(rows)

def _strip_trivariate(cube: list[list[list[int]]]) -> list[list[list[int]]]:
    if not cube:
        return [[[0]]]
    cube = [[[mod(c) for c in row] for row in plane] for plane in cube]
    while len(cube) > 1 and all((coef == 0 for row in cube[-1] for coef in row)):
        cube.pop()
    depth = max((len(mat) for mat in cube))
    width = max((len(row) for mat in cube for row in mat))
    out: list[list[list[int]]] = []
    for mat in cube:
        padded = [_zero_pad_row(row, width) for row in mat]
        padded += [[0] * width] * (depth - len(padded))
        out.append(padded)
    while width > 1 and all((out[a][b][width - 1] == 0 for a in range(len(out)) for b in range(len(out[a])))):
        width -= 1
        out = [plane[:width] for plane in out]
    return out if out else [[[0]]]

class TriPoly:
    __slots__ = ('coeffs',)

    def __init__(self, coeffs: list[list[list[int]]]) -> None:
        self.coeffs = _strip_trivariate(coeffs)

    def _deg_axis(self, axis: int) -> int:
        if axis == 0:
            return len(self.coeffs) - 1
        if axis == 1:
            return max((len(plane) for plane in self.coeffs)) - 1
        return max((len(row) for plane in self.coeffs for row in plane)) - 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TriPoly):
            return NotImplemented
        return self.coeffs == other.coeffs

    def eval(self, x: int, y: int, z: int) -> int:
        x, y, z = (mod(x), mod(y), mod(z))
        acc = 0
        for a, plane in enumerate(self.coeffs):
            xpow = _x_pow(x, a)
            for b, row in enumerate(plane):
                ypow = _x_pow(y, b)
                zpow = 1
                for c, coef in enumerate(row):
                    if coef:
                        acc = add(acc, mul(coef, mul(xpow, mul(ypow, zpow))))
                    zpow = mul(zpow, z)
        return acc

    def fix_x(self, x: int) -> BiPoly:
        x = mod(x)
        dy, dz = (self._deg_axis(1), self._deg_axis(2))
        rows = [[0] * (dz + 1) for _ in range(dy + 1)]
        for a, plane in enumerate(self.coeffs):
            xpow = _x_pow(x, a)
            for b, row in enumerate(plane):
                for c, coef in enumerate(row):
                    if coef:
                        rows[b][c] = add(rows[b][c], mul(coef, xpow))
        return BiPoly(rows)

    def fix_y(self, y: int) -> BiPoly:
        y = mod(y)
        dx, dz = (self._deg_axis(0), self._deg_axis(2))
        rows = [[0] * (dz + 1) for _ in range(dx + 1)]
        for a, plane in enumerate(self.coeffs):
            for b, row in enumerate(plane):
                ypow = _x_pow(y, b)
                for c, coef in enumerate(row):
                    if coef:
                        rows[a][c] = add(rows[a][c], mul(coef, ypow))
        return BiPoly(rows)

    def fix_z(self, z: int) -> BiPoly:
        z = mod(z)
        dx, dy = (self._deg_axis(0), self._deg_axis(1))
        rows = [[0] * (dy + 1) for _ in range(dx + 1)]
        for a, plane in enumerate(self.coeffs):
            for b, row in enumerate(plane):
                zpow = 1
                for c, coef in enumerate(row):
                    if coef:
                        rows[a][b] = add(rows[a][b], mul(coef, zpow))
                    zpow = mul(zpow, z)
        return BiPoly(rows)

    def fix_xy(self, x: int, y: int) -> UniPoly:
        x, y = (mod(x), mod(y))
        dz = self._deg_axis(2)
        uni = [0] * (dz + 1)
        for a, plane in enumerate(self.coeffs):
            xpow = _x_pow(x, a)
            for b, row in enumerate(plane):
                ypow = _x_pow(y, b)
                xy = mul(xpow, ypow)
                for c, coef in enumerate(row):
                    if coef:
                        uni[c] = add(uni[c], mul(coef, xy))
        return UniPoly(uni)

    @classmethod
    def random(cls, deg_x: int, deg_y: int, deg_z: int, constant: int | None=None, rng: Random | None=None) -> TriPoly:
        if deg_x < 0 or deg_y < 0 or deg_z < 0:
            raise ValueError('degrees must be non-negative')
        cube = [[[rand_field(rng) for _ in range(deg_z + 1)] for _ in range(deg_y + 1)] for _ in range(deg_x + 1)]
        if constant is not None:
            cube[0][0][0] = mod(constant)
        return cls(cube)

def eval_poly(coeffs: list[int], x: int) -> int:
    return UniPoly(coeffs).eval(x)

def interpolate_at_zero(ys: list[int], xs: list[int]) -> int:
    if len(ys) != len(xs):
        raise ValueError('xs and ys length mismatch')
    points = list(zip(xs, ys))
    return UniPoly.interpolate(points).coeff(0)