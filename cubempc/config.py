from __future__ import annotations
from dataclasses import dataclass, field
from cubempc.field import P as FIELD_PRIME
from cubempc.host_layout import HostLayout
DEFAULT_PRIME: int = FIELD_PRIME

def default_threshold(n: int) -> int:
    return (n - 1) // 4

def validate_party_count(n: int, t: int | None=None) -> int:
    if n < 1:
        raise ValueError(f'n must be positive, got {n}')
    threshold = default_threshold(n) if t is None else t
    if threshold < 0:
        raise ValueError(f't must be non-negative, got {threshold}')
    if n < 4 * threshold + 1:
        raise ValueError(f'need n >= 4t + 1, got n={n}, t={threshold} (requires n >= {4 * threshold + 1})')
    return threshold

@dataclass
class MPCConfig:
    n: int
    t: int
    p: int = FIELD_PRIME
    base_port: int = 18000
    hosts: tuple[str, ...] | None = None

    @classmethod
    def create(cls, n: int, *, t: int | None=None, p: int=FIELD_PRIME, base_port: int=18000, hosts: tuple[str, ...] | None=None) -> MPCConfig:
        threshold = validate_party_count(n, t)
        return cls(n=n, t=threshold, p=p, base_port=base_port, hosts=hosts)

    def host_layout(self) -> HostLayout:
        if self.hosts:
            return HostLayout(self.hosts)
        return HostLayout.localhost()

    def node_port(self, rank: int) -> int:
        if rank < 0 or rank >= self.n:
            raise ValueError(f'rank {rank} out of range [0, {self.n})')
        return self.base_port + rank

@dataclass
class LayerState:
    layer: int
    values: dict = field(default_factory=dict)
    protocol_states: dict = field(default_factory=dict)

@dataclass(frozen=True)
class ExperimentConfig:
    n: int
    num_layers: int
    t: int
    prime: int = FIELD_PRIME
    host: str = '127.0.0.1'
    hosts: tuple[str, ...] | None = None
    base_port: int = 19000
    byzantine_ranks: frozenset[int] = frozenset()

    @classmethod
    def create(cls, n: int, num_layers: int, *, t: int | None=None, prime: int=FIELD_PRIME, host: str='127.0.0.1', hosts: tuple[str, ...] | None=None, base_port: int=19000, byzantine_ranks: frozenset[int] | None=None) -> ExperimentConfig:
        threshold = validate_party_count(n, t)
        if num_layers < 1:
            raise ValueError(f'num_layers must be >= 1, got {num_layers}')
        if hosts is not None and len(hosts) == 0:
            raise ValueError('hosts must not be empty')
        ranks = byzantine_ranks or frozenset()
        for r in ranks:
            if r < 0 or r >= n:
                raise ValueError(f'byzantine rank {r} out of range [0, {n})')
        return cls(n=n, num_layers=num_layers, t=threshold, prime=prime, host=host, hosts=hosts, base_port=base_port, byzantine_ranks=ranks)

    def node_port(self, rank: int) -> int:
        if rank < 0 or rank >= self.n:
            raise ValueError(f'rank {rank} out of range [0, {self.n})')
        return self.base_port + rank

    def to_mpc_config(self) -> MPCConfig:
        return MPCConfig(n=self.n, t=self.t, p=self.prime, base_port=self.base_port, hosts=self.hosts or (self.host,))