from __future__ import annotations
from random import Random
from cubempc.circuits.circuit import Circuit, Gate, validate_circuit

def make_synthetic_layered_circuit(depth: int, width: int, mul_ratio: float, seed: int, num_inputs: int | None=None) -> Circuit:
    if depth < 1:
        raise ValueError(f'depth must be >= 1, got {depth}')
    if width < 1:
        raise ValueError(f'width must be >= 1, got {width}')
    if num_inputs is None:
        num_inputs = width
    if num_inputs < 1:
        raise ValueError(f'num_inputs must be >= 1, got {num_inputs}')
    if not 0.0 <= mul_ratio <= 1.0:
        raise ValueError(f'mul_ratio must be in [0, 1], got {mul_ratio}')
    rng = Random(seed)
    input_names = [f'in_{i}' for i in range(num_inputs)]
    gates: list[Gate] = [Gate(gid=name, op='input', depth=0) for name in input_names]
    current_wires = list(input_names)
    mul_count = round(width * mul_ratio)
    for layer in range(1, depth + 1):
        ops = ['mul'] * mul_count + ['add'] * (width - mul_count)
        rng.shuffle(ops)
        new_wires: list[str] = []
        for g_idx, op in enumerate(ops):
            gid = f'L{layer}_w{g_idx}'
            gates.append(Gate(gid=gid, op=op, in1=rng.choice(current_wires), in2=rng.choice(current_wires), depth=layer))
            new_wires.append(gid)
        current_wires = new_wires
    circuit = Circuit(inputs=input_names, gates=gates, output=current_wires[0])
    validate_circuit(circuit)
    return circuit

def generate_layered_arithmetic_circuit(num_inputs: int, depth: int, width: int, mul_ratio: float, rng: Random | None=None) -> Circuit:
    if num_inputs < 1:
        raise ValueError(f'num_inputs must be >= 1, got {num_inputs}')
    if depth < 1:
        raise ValueError(f'depth must be >= 1, got {depth}')
    if width < 1:
        raise ValueError(f'width must be >= 1, got {width}')
    if not 0.0 <= mul_ratio <= 1.0:
        raise ValueError(f'mul_ratio must be in [0, 1], got {mul_ratio}')
    gen = rng if rng is not None else Random(0)
    input_names = [f'in_{i}' for i in range(num_inputs)]
    gates: list[Gate] = [Gate(gid=name, op='input', depth=0) for name in input_names]
    current_wires = list(input_names)
    for layer in range(1, depth + 1):
        new_wires: list[str] = []
        for g_idx in range(width):
            a = gen.choice(current_wires)
            b = gen.choice(current_wires)
            use_mul = gen.random() < mul_ratio
            gid = f'L{layer}_w{g_idx}'
            if use_mul:
                gates.append(Gate(gid=gid, op='mul', in1=a, in2=b, depth=layer))
            else:
                gates.append(Gate(gid=gid, op='add', in1=a, in2=b, depth=layer))
            new_wires.append(gid)
        current_wires = new_wires
    circuit = Circuit(inputs=input_names, gates=gates, output=current_wires[0])
    validate_circuit(circuit)
    return circuit

def generate_add_chain(num_inputs: int) -> Circuit:
    if num_inputs < 1:
        raise ValueError('num_inputs must be >= 1')
    input_names = [f'in_{i}' for i in range(num_inputs)]
    gates: list[Gate] = [Gate(gid=name, op='input', depth=0) for name in input_names]
    acc = input_names[0]
    for i in range(1, num_inputs):
        gid = f'add_{i}'
        gates.append(Gate(gid=gid, op='add', in1=acc, in2=input_names[i], depth=i))
        acc = gid
    circuit = Circuit(inputs=input_names, gates=gates, output=acc)
    validate_circuit(circuit)
    return circuit

def generate_multiplication_depth(depth: int) -> Circuit:
    if depth < 1:
        raise ValueError('depth must be >= 1')
    input_names = ['in_0', 'in_1']
    gates: list[Gate] = [Gate(gid='in_0', op='input', depth=0), Gate(gid='in_1', op='input', depth=0)]
    left, right = ('in_0', 'in_1')
    for layer in range(1, depth + 1):
        gid = f'mul_{layer}'
        gates.append(Gate(gid=gid, op='mul', in1=left, in2=right, depth=layer))
        left = gid
        right = gid
    circuit = Circuit(inputs=input_names, gates=gates, output=left)
    validate_circuit(circuit)
    return circuit