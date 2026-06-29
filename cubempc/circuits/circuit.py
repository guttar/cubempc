from __future__ import annotations
from dataclasses import dataclass
from cubempc.field import add, mod, mul

@dataclass
class Gate:
    gid: str
    op: str
    in1: str | None = None
    in2: str | None = None
    const: int | None = None
    depth: int = 0

@dataclass
class Circuit:
    inputs: list[str]
    gates: list[Gate]
    output: str

def evaluate_plain(circuit: Circuit, inputs: dict[str, int]) -> int:
    values: dict[str, int] = {}
    for name in circuit.inputs:
        if name not in inputs:
            raise KeyError(f'missing input wire {name!r}')
        values[name] = mod(inputs[name])
    for gate in circuit.gates:
        if gate.op == 'input':
            if gate.gid not in values:
                if gate.gid not in inputs:
                    raise KeyError(f'input gate {gate.gid!r} has no value')
                values[gate.gid] = mod(inputs[gate.gid])
            continue
        if gate.op == 'add':
            if gate.in1 is None or gate.in2 is None:
                raise ValueError(f'add gate {gate.gid!r} needs in1, in2')
            values[gate.gid] = add(values[gate.in1], values[gate.in2])
        elif gate.op == 'mul':
            if gate.in1 is None or gate.in2 is None:
                raise ValueError(f'mul gate {gate.gid!r} needs in1, in2')
            values[gate.gid] = mul(values[gate.in1], values[gate.in2])
        elif gate.op == 'cmul':
            if gate.in1 is None or gate.const is None:
                raise ValueError(f'cmul gate {gate.gid!r} needs in1, const')
            values[gate.gid] = mul(mod(gate.const), values[gate.in1])
        else:
            raise ValueError(f'unknown gate op {gate.op!r} on {gate.gid!r}')
    if circuit.output not in values:
        raise ValueError(f'output wire {circuit.output!r} was never defined')
    return values[circuit.output]

def count_gates_by_op(circuit: Circuit) -> dict[str, int]:
    counts: dict[str, int] = {'add': 0, 'mul': 0, 'cmul': 0}
    for gate in circuit.gates:
        if gate.op in counts:
            counts[gate.op] += 1
    return counts

def validate_circuit(circuit: Circuit) -> None:
    if not circuit.inputs:
        raise ValueError('circuit must have at least one input')
    if circuit.output not in circuit.inputs and (not any((g.gid == circuit.output for g in circuit.gates))):
        pass
    defined: set[str] = set()
    input_set = set(circuit.inputs)
    if len(input_set) != len(circuit.inputs):
        raise ValueError('duplicate names in circuit.inputs')
    for gate in circuit.gates:
        if gate.op == 'input':
            if gate.gid not in input_set:
                raise ValueError(f'input gate {gate.gid!r} not listed in circuit.inputs')
            if gate.gid in defined:
                raise ValueError(f'duplicate input gate {gate.gid!r}')
            defined.add(gate.gid)
            continue
        for dep in (gate.in1, gate.in2):
            if dep is not None and dep not in defined:
                raise ValueError(f'gate {gate.gid!r} ({gate.op}) references undefined wire {dep!r}')
        if gate.op == 'add':
            if gate.in1 is None or gate.in2 is None:
                raise ValueError(f'add gate {gate.gid!r} missing operands')
        elif gate.op == 'mul':
            if gate.in1 is None or gate.in2 is None:
                raise ValueError(f'mul gate {gate.gid!r} missing operands')
        elif gate.op == 'cmul':
            if gate.in1 is None or gate.const is None:
                raise ValueError(f'cmul gate {gate.gid!r} missing in1 or const')
        else:
            raise ValueError(f'unknown op {gate.op!r}')
        if gate.gid in defined:
            raise ValueError(f'duplicate wire id {gate.gid!r}')
        defined.add(gate.gid)
    if circuit.output not in defined:
        raise ValueError(f'output wire {circuit.output!r} is not defined')