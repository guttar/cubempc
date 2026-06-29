from __future__ import annotations

_LOCAL_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1'})


class HostLayout:
    __slots__ = ('_hosts',)

    def __init__(self, hosts: tuple[str, ...]) -> None:
        if not hosts:
            raise ValueError('hosts must not be empty')
        self._hosts = hosts

    @classmethod
    def from_list(cls, hosts: list[str]) -> HostLayout:
        cleaned = [host.strip() for host in hosts if host.strip()]
        if not cleaned:
            raise ValueError('hosts must not be empty')
        return cls(tuple(cleaned))

    @classmethod
    def localhost(cls) -> HostLayout:
        return cls(('127.0.0.1',))

    @classmethod
    def parse(cls, raw: str | None) -> HostLayout:
        if raw is None or not raw.strip():
            return cls.localhost()
        return cls.from_list([part.strip() for part in raw.split(',') if part.strip()])

    @property
    def hosts(self) -> tuple[str, ...]:
        return self._hosts

    def machine_count(self) -> int:
        return len(self._hosts)

    def machine_id(self, rank: int) -> int:
        return rank % len(self._hosts)

    def host_for_rank(self, rank: int) -> str:
        return self._hosts[self.machine_id(rank)]

    def is_local_only(self) -> bool:
        return all(host in _LOCAL_HOSTS for host in self._hosts)

    def is_distributed(self) -> bool:
        return not self.is_local_only() or len(set(self._hosts)) > 1
