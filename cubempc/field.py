from __future__ import annotations
import secrets
from random import Random
P: int = 2 ** 61 - 1
FIELD_PRIME: int = P

def mod(x: int) -> int:
    return x % P

def add(a: int, b: int) -> int:
    return (a + b) % P

def sub(a: int, b: int) -> int:
    return (a - b) % P

def neg(a: int) -> int:
    return -a % P

def mul(a: int, b: int) -> int:
    return a * b % P

def pow_mod(a: int, e: int) -> int:
    return pow(mod(a), e, P)

def inv(a: int) -> int:
    a = mod(a)
    if a == 0:
        raise ZeroDivisionError
    return pow(a, P - 2, P)

def div(a: int, b: int) -> int:
    return mul(a, inv(b))

def rand_field(rng: Random | None=None) -> int:
    if rng is not None:
        return rng.randrange(P)
    return secrets.randbelow(P)