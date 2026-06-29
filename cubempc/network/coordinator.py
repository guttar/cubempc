from __future__ import annotations
import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any
from cubempc.config import ExperimentConfig
from cubempc.messages import Message
from cubempc.metrics import Metrics
from cubempc.network.node import NodeProcess
logger = logging.getLogger(__name__)

@dataclass
class ProcessHandle:
    rank: int
    process: mp.Process

@dataclass
class NodeResult:
    rank: int
    pid: int
    p2p_received: int
    broadcast_received: int
    metrics: dict[str, int]

def _metrics_to_dict(metrics: Metrics) -> dict[str, int]:
    return {'serialized_p2p_bytes': metrics.serialized_p2p_bytes, 'serialized_broadcast_bytes': metrics.serialized_broadcast_bytes, 'message_count': metrics.message_count, 'physical_bytes': metrics.physical_bytes, 'serialized_total_bytes': metrics.total_serialized_bytes()}

def _metrics_from_dict(data: dict[str, int]) -> Metrics:
    return Metrics(serialized_p2p_bytes=data.get('serialized_p2p_bytes', data.get('p2p_bytes', 0)), serialized_broadcast_bytes=data.get('serialized_broadcast_bytes', data.get('broadcast_bytes', 0)), message_count=data['message_count'], physical_bytes=data.get('physical_bytes', 0))

def join_worker_processes(processes: list[mp.Process], *, join_timeout: float=15.0, terminate_timeout: float=5.0) -> None:
    deadline = time.time() + join_timeout
    for proc in processes:
        remaining = max(0.0, deadline - time.time())
        proc.join(timeout=remaining)
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=terminate_timeout)
    bad = [proc.exitcode for proc in processes if proc.exitcode not in (0, None)]
    if bad:
        raise RuntimeError(f'worker process bad exit codes: {bad}')

async def _run_ping_node(rank: int, n: int, base_port: int, instance_id: str, startup_delay: float, hosts: tuple[str, ...] | None=None) -> NodeResult:
    config = ExperimentConfig.create(n=n, num_layers=1, base_port=base_port, hosts=hosts)
    node = NodeProcess(rank, n, base_port, config)
    await node.start_server()
    await asyncio.sleep(startup_delay)
    dst = (rank + 1) % n
    await node.send_p2p(dst, Message(src_rank=rank, dst_rank=dst, src_layer=0, dst_layer=1, protocol='ping', instance_id=instance_id, msg_type='ping', channel='p2p', payload={'text': 'ping'}))
    await node.broadcast(Message(src_rank=rank, dst_rank=None, src_layer=0, dst_layer=1, protocol='ping', instance_id=instance_id, msg_type='hello', channel='broadcast', payload={'text': 'hello'}))
    p2p_filter = lambda m: m.protocol == 'ping' and m.msg_type == 'ping' and (m.channel == 'p2p') and (m.instance_id == instance_id)
    bcast_filter = lambda m: m.protocol == 'ping' and m.msg_type == 'hello' and (m.channel == 'broadcast') and (m.instance_id == instance_id)
    p2ps = await node.recv_until(p2p_filter, expected_count=1, timeout=15.0)
    hellos = await node.recv_until(bcast_filter, expected_count=n, timeout=15.0)
    await node.stop_server()
    return NodeResult(rank=rank, pid=0, p2p_received=len(p2ps), broadcast_received=len(hellos), metrics=_metrics_to_dict(node.metrics))

def _node_entry(rank: int, n: int, base_port: int, instance_id: str, startup_delay: float, hosts: tuple[str, ...] | None, out_queue: mp.Queue) -> None:
    try:
        result = asyncio.run(_run_ping_node(rank, n, base_port, instance_id, startup_delay, hosts))
        result.pid = mp.current_process().pid
        out_queue.put(result)
    except Exception as exc:
        out_queue.put({'rank': rank, 'error': str(exc), 'pid': mp.current_process().pid})

class Coordinator:

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._processes: list[ProcessHandle] = []
        self._results: list[NodeResult] = []

    def spawn_nodes(self, *, protocol: str='ping', startup_delay: float=1.0) -> list[ProcessHandle]:
        if self._processes:
            raise RuntimeError('nodes already spawned; refusing duplicate spawn')
        if protocol != 'ping':
            raise ValueError(f'unsupported protocol: {protocol}')
        instance_id = str(uuid.uuid4())
        out_queue: mp.Queue = mp.Queue()
        for rank in range(self.config.n):
            proc = mp.Process(target=_node_entry, args=(rank, self.config.n, self.config.base_port, instance_id, startup_delay, self.config.hosts, out_queue), name=f'cubempc-node-{rank}')
            proc.start()
            self._processes.append(ProcessHandle(rank=rank, process=proc))
            logger.info('spawned rank %s pid %s', rank, proc.pid)
        self._results = self._collect_results(out_queue, self.config.n, timeout=60.0)
        return list(self._processes)

    def _collect_results(self, out_queue: mp.Queue, n: int, *, timeout: float) -> list[NodeResult]:
        deadline = time.time() + timeout
        raw: list[Any] = []
        while len(raw) < n and time.time() < deadline:
            try:
                raw.append(out_queue.get(timeout=0.5))
            except Exception:
                continue
        results: list[NodeResult] = []
        for item in raw:
            if isinstance(item, NodeResult):
                results.append(item)
            elif isinstance(item, dict) and 'error' in item:
                raise RuntimeError(f'node {item.get('rank')} failed: {item['error']}')
        if len(results) < n:
            raise RuntimeError(f'expected {n} node results, got {len(results)}')
        return sorted(results, key=lambda r: r.rank)

    def aggregate_metrics(self) -> Metrics:
        total = Metrics()
        for result in self._results:
            m = _metrics_from_dict(result.metrics)
            total.merge(m)
        return total

    def metrics_json(self) -> dict[str, Any]:
        aggregated = self.aggregate_metrics()
        return {'n': self.config.n, 'base_port': self.config.base_port, 'per_rank': [asdict(r) for r in self._results], 'aggregated': _metrics_to_dict(aggregated)}

    def terminate_all(self, grace_sec: float=2.0) -> None:
        for handle in self._processes:
            if handle.process.is_alive():
                handle.process.terminate()
        deadline = time.time() + grace_sec
        for handle in self._processes:
            while handle.process.is_alive() and time.time() < deadline:
                time.sleep(0.05)
        for handle in self._processes:
            if handle.process.is_alive():
                handle.process.kill()
            handle.process.join(timeout=1.0)
        self._processes.clear()

    def __enter__(self) -> Coordinator:
        self.spawn_nodes()
        return self

    def __exit__(self, *exc: object) -> None:
        self.terminate_all()

def run_ping_benchmark(n: int, base_port: int, *, startup_delay: float=1.0, hosts: tuple[str, ...] | None=None) -> dict[str, Any]:
    config = ExperimentConfig.create(n=n, num_layers=1, base_port=base_port, hosts=hosts)
    coord = Coordinator(config)
    try:
        coord.spawn_nodes(protocol='ping', startup_delay=startup_delay)
        return coord.metrics_json()
    finally:
        coord.terminate_all()

def main(argv: list[str] | None=None) -> None:
    parser = argparse.ArgumentParser(description='CUBE MPC network coordinator')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--base-port', type=int, default=19000)
    parser.add_argument('--protocol', type=str, default='ping')
    parser.add_argument('--startup-delay', type=float, default=1.0)
    parser.add_argument('--hosts', type=str, default=None, help='comma-separated hostnames/IPs for rank placement')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    hosts = tuple(part.strip() for part in args.hosts.split(',') if part.strip()) if args.hosts else None
    if args.protocol == 'ping':
        report = run_ping_benchmark(args.n, args.base_port, startup_delay=args.startup_delay, hosts=hosts)
        print(json.dumps(report, indent=2))
    else:
        raise SystemExit(f'unsupported protocol: {args.protocol}')
if __name__ == '__main__':
    main()