import itertools

import networkx as nx
from qiskit.circuit import QuantumCircuit, QuantumRegister

from qtpu.instructions import WireCut, InstanceGate
from qtpu.virtual_gates import VirtualMove, VirtualBinaryGate, VIRTUAL_GATE_GENERATORS
from qtpu.tensor import HybridTensorNetwork, QuantumTensor, ClassicalTensor


def insert_cuts(
    circuit: QuantumCircuit, gate_cuts: set[int], wire_cuts: set[tuple[int, int]]
) -> QuantumCircuit:
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    for i, instr in enumerate(circuit):
        op, qubits = instr.operation, instr.qubits

        if i in gate_cuts:
            op = VIRTUAL_GATE_GENERATORS[op.name](op.params)

        new_circuit.append(op, qubits, instr.clbits)

        for j in range(len(qubits)):
            if (i, j) in wire_cuts:
                new_circuit.append(WireCut(), [qubits[j]], [])
    return new_circuit


def circuit_to_hybrid_tn(circuit: QuantumCircuit) -> HybridTensorNetwork:
    circuit = fragment(circuit)
    ctensors = _extract_classical_tensors(circuit)

    circuit = _decompose_virtual_gates(circuit)

    qtensors = []
    for qreg in circuit.qregs:
        qtensors.append(QuantumTensor(_circuit_on_qreg(circuit, qreg)))

    return HybridTensorNetwork(qtensors, ctensors)


def wire_cuts_to_move(circuit: QuantumCircuit) -> QuantumCircuit:
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    qubit_mapping = {}

    def find_qubit(qubit):
        while qubit in qubit_mapping:
            qubit = qubit_mapping[qubit]
        return qubit

    ctr = 0
    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits

        qubits = [find_qubit(qubit) for qubit in qubits]

        if isinstance(op, WireCut):
            qreg = QuantumRegister(1, f"wc{ctr}")
            ctr += 1
            circuit.add_register(qreg)
            qubit_mapping[qubits[0]] = qreg[0]
            circuit.append(VirtualMove(), [qubits[0], qreg[0]])
            continue

        circuit.append(op, qubits, clbits)

    return new_circuit


def fragment(circuit: QuantumCircuit) -> QuantumCircuit:
    ccs = nx.connected_components(_qubit_graph(circuit))

    fragments = [QuantumRegister(len(cc), name=f"f{i}") for i, cc in enumerate(ccs)]
    sorted_ccs = [sorted(cc, key=lambda q: circuit.qubits.index(q)) for cc in ccs]

    qubit_mapping = {}
    for fragment, cc in zip(fragments, sorted_ccs):
        for i, qubit in enumerate(cc):
            qubit_mapping[qubit] = fragment[i]

    new_circuit = QuantumCircuit(*fragments, *circuit.cregs)
    for instr in circuit:
        qubits = [qubit_mapping[qubit] for qubit in instr.qubits]
        new_circuit.append(instr.operation, qubits, instr.clbits)
    return new_circuit


def _qubit_graph(circuit: QuantumCircuit) -> nx.Graph:
    graph = nx.Graph()
    for instr in circuit:
        if instr.operation.name == "barrier":
            continue

        qubits = instr.qubits
        for qubit1, qubit2 in itertools.combinations(qubits, 2):
            graph.add_edge(qubit1, qubit2)
    return graph


def _extract_classical_tensors(circuit: QuantumCircuit) -> list[ClassicalTensor]:
    ctensors = []
    for instr in circuit:
        op, qubits = instr.operation, instr.qubits
        if isinstance(op, VirtualBinaryGate):
            reg1 = circuit.find_bit(qubits[0]).registers[0][0]
            reg2 = circuit.find_bit(qubits[1]).registers[0][0]
            if reg1 == reg2:
                ct = ClassicalTensor(op.coefficients_1d(), [op.idx])
                ctensors.append(ct)
            else:
                ct = ClassicalTensor(
                    op.coefficients_2d(), [f"{op.idx}_0", f"{op.idx}_1"]
                )
                ctensors.append(ct)
    return ctensors


def _decompose_virtual_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        if isinstance(op, VirtualBinaryGate):
            reg1 = circuit.find_bit(qubits[0]).registers[0][0]
            reg2 = circuit.find_bit(qubits[1]).registers[0][0]
            if reg1 == reg2:
                ig = InstanceGate(2, op.idx, op.instantiations()), qubits, clbits
                new_circuit.append(ig)
            else:
                ig1 = InstanceGate(1, f"{op.idx}_0", op.instances_q0())
                ig2 = InstanceGate(1, f"{op.idx}_1", op.instances_q1())
                new_circuit.append(ig1, [qubits[0]], [])
                new_circuit.append(ig2, [qubits[1]], [])
            continue

        new_circuit.append(op, qubits, clbits)
    return new_circuit


def _circuit_on_qreg(circuit: QuantumCircuit, qreg: QuantumRegister) -> QuantumCircuit:
    new_circuit = QuantumCircuit(qreg, *circuit.cregs)
    for instr in circuit:
        qubits = instr.operation
        intersection = set(qubits) & set(qreg)

        if len(intersection) == 0:
            continue

        elif len(intersection) == len(qubits):
            new_circuit.append(instr)

        else:
            raise ValueError("Operation spans multiple qregs.")

    return new_circuit
