from __future__ import annotations
import socket
import time
DEFAULT_PORT_BLOCK_BUFFER = 8
PORT_STRIDE = 1000

def port_block_span(n: int, *, buffer: int=DEFAULT_PORT_BLOCK_BUFFER) -> int:
    if n < 1:
        raise ValueError(f'n must be >= 1, got {n}')
    if buffer < 0:
        raise ValueError(f'buffer must be >= 0, got {buffer}')
    return n + buffer

def _ports_bindable(host: str, base: int, span: int) -> bool:
    sockets: list[socket.socket] = []
    try:
        for port in range(base, base + span):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()

def find_free_base_port(n: int, *, buffer: int=DEFAULT_PORT_BLOCK_BUFFER, host: str='127.0.0.1', candidate: int | None=None, used_ports: set[int] | None=None) -> int:
    span = port_block_span(n, buffer=buffer)
    used = used_ports if used_ports is not None else set()
    base = candidate
    for attempt in range(200):
        if base is None:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, 0))
                base = probe.getsockname()[1]
            finally:
                probe.close()
        block = set(range(base, base + span))
        if base + span < 65535 and (not block & used) and _ports_bindable(host, base, span):
            if used_ports is not None:
                used_ports.update(block)
            return base
        base += PORT_STRIDE
        if attempt + 1 < 200:
            time.sleep(0.05)
    raise RuntimeError(f'could not allocate {span} consecutive free ports on {host} (n={n}, buffer={buffer}, candidate={candidate})')