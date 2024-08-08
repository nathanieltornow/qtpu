import itertools

import quimb.tensor as qtn
import networkx as nx
from qiskit.circuit import QuantumCircuit, QuantumRegister

from circuit_knitting.cutting.instructions import CutWire, Move
from circuit_knitting.cutting.qpd import (
    QPDBasis,
    QPDMeasure,
    TwoQubitQPDGate,
    SingleQubitQPDGate,
)

from qtpu.helpers import remove_barriers
from qtpu.instructions import InstanceGate
from qtpu.tensor import QuantumTensor, HybridTensorNetwork, wire_tensor


def insert_cuts(
    circuit: QuantumCircuit, gate_cuts: set[int], wire_cuts: set[tuple[int, int]]
) -> QuantumCircuit:
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    for i, instr in enumerate(circuit):
        op, qubits = instr.operation, instr.qubits

        if i in gate_cuts:
            op = TwoQubitQPDGate.from_instruction(op)

        new_circuit.append(op, qubits, instr.clbits)

        for j in range(len(qubits)):
            if (i, j) in wire_cuts:
                new_circuit.append(CutWire(), [qubits[j]], [])
    return new_circuit


def circuit_to_hybrid_tn(circuit: QuantumCircuit) -> HybridTensorNetwork:
    circuit = fragment(remove_barriers(circuit))
    ctensors = _extract_qpd_tensors(circuit)

    circuit = _decompose_virtual_gates(circuit)

    qtensors = []
    for qreg in circuit.qregs:
        qtensors.append(QuantumTensor(_circuit_on_qreg(circuit, qreg)))

    return HybridTensorNetwork(qtensors, ctensors)


def cuts_to_moves(circuit: QuantumCircuit) -> QuantumCircuit:

    qubit_mapping = {}

    def _find_qubit(qubit):
        while qubit in qubit_mapping:
            qubit = qubit_mapping[qubit]
        return qubit

    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    ctr = 0
    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        qubits = [_find_qubit(qubit) for qubit in qubits]

        if isinstance(op, CutWire):
            new_reg = QuantumRegister(1, name=f"cut_{ctr}")
            new_circuit.add_register(new_reg)
            qubit_mapping[qubits[0]] = new_reg[0]
            ctr += 1
            qubits += [new_reg[0]]
            op = TwoQubitQPDGate.from_instruction(Move())

        new_circuit.append(op, qubits, clbits)
    return new_circuit


def fragment(circuit: QuantumCircuit) -> QuantumCircuit:
    ccs = list(nx.connected_components(_qubit_graph(circuit)))

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
    graph.add_nodes_from(circuit.qubits)
    for instr in circuit:
        if instr.operation.name == "barrier" or isinstance(
            instr.operation, TwoQubitQPDGate
        ):
            continue

        qubits = instr.qubits
        for qubit1, qubit2 in itertools.combinations(qubits, 2):
            graph.add_edge(qubit1, qubit2)
    return graph


def _extract_qpd_tensors(circuit: QuantumCircuit) -> list[qtn.Tensor]:
    qpd_tensors = []
    qpd_ctr = 0
    for instr in circuit:
        op, qubits = instr.operation, instr.qubits

        assert not isinstance(op, SingleQubitQPDGate)

        if isinstance(op, TwoQubitQPDGate):
            ct = qtn.Tensor(op.basis.coeffs, inds=[str(qpd_ctr)])
            if (
                op.label == "cut_move"
                and circuit.find_bit(qubits[0]).registers[0][0]
                != circuit.find_bit(qubits[1]).registers[0][0]
            ):
                ct = wire_tensor(str(qpd_ctr))

            qpd_tensors.append(ct)
            qpd_ctr += 1
    return qpd_tensors


def _decompose_virtual_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    qpd_ctr = 0
    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        if isinstance(op, TwoQubitQPDGate):
            reg1 = circuit.find_bit(qubits[0]).registers[0][0]
            reg2 = circuit.find_bit(qubits[1]).registers[0][0]
            if reg1 == reg2:
                ig = _qpd_to_instance_gate(op.basis, str(qpd_ctr))
                new_circuit.append(ig, qubits, [])
            else:
                ig1, ig2 = _qpd_to_instance_gate_2qubit(op.basis, str(qpd_ctr))
                if op.label == "cut_move":
                    ig1, ig2 = _move_instance_gates(str(qpd_ctr))
                new_circuit.append(ig1, [qubits[0]], [])
                new_circuit.append(ig2, [qubits[1]], [])
            qpd_ctr += 1
            continue

        new_circuit.append(op, qubits, clbits)
    return new_circuit


def _qpd_to_instance_gate(basis: QPDBasis, index: str) -> InstanceGate:
    instances = []
    for map in basis.maps:
        circuit = QuantumCircuit(2, 1)
        for op in map[0]:
            if isinstance(op, QPDMeasure):
                circuit.measure(0, 0)
                continue
            # if op.name == "reset":
            #     continue
            circuit.append(op, [0], [])
        for op in map[1]:
            if isinstance(op, QPDMeasure):
                circuit.measure(1, 0)
                continue
            # if op.name == "reset":
            #     continue
            circuit.append(op, [1], [])
        instances.append(circuit)
    return InstanceGate(2, index, instances)


def _qpd_to_instance_gate_2qubit(
    qpd_basis: QPDBasis, index: str
) -> tuple[InstanceGate, InstanceGate]:
    instances = ([], [])
    for map in qpd_basis.maps:
        c1 = QuantumCircuit(1, 1)
        c2 = QuantumCircuit(1, 1)
        for op in map[0]:
            if isinstance(op, QPDMeasure):
                c1.measure(0, 0)
                continue
            # if op.name == "reset":
            # continue
            c1.append(op, [0], [])
        for op in map[1]:
            if isinstance(op, QPDMeasure):
                c2.measure(0, 0)
                continue
            # if op.name == "reset":
            #     continue
            c2.append(op, [0], [])
        instances[0].append(c1)
        instances[1].append(c2)

    return (
        InstanceGate(1, index, instances[0]),
        InstanceGate(1, index, instances[1]),
    )


def _circuit_on_qreg(circuit: QuantumCircuit, qreg: QuantumRegister) -> QuantumCircuit:
    new_circuit = QuantumCircuit(qreg, *circuit.cregs)
    for instr in circuit:
        qubits = instr.qubits
        intersection = set(qubits) & set(qreg)

        if len(intersection) == 0:
            continue

        elif len(intersection) == len(qubits):
            new_circuit.append(instr)

        else:
            raise ValueError("Operation spans multiple qregs.")

    return new_circuit


def _move_instance_gates(ind: str) -> tuple[InstanceGate, InstanceGate]:
    i = QuantumCircuit(1, 1)
    i.reset(0)

    z = QuantumCircuit(1, 1)
    z.measure(0, 0)
    z.reset(0)

    x = QuantumCircuit(1, 1)
    x.h(0)
    x.measure(0, 0)
    x.reset(0)

    y = QuantumCircuit(1, 1)
    y.sx(0)
    y.measure(0, 0)
    y.reset(0)

    zero = QuantumCircuit(1, 1)
    zero.reset(0)

    one = QuantumCircuit(1, 1)
    one.reset(0)
    one.x(0)

    plus = QuantumCircuit(1, 1)
    plus.reset(0)
    plus.h(0)

    iplus = QuantumCircuit(1, 1)
    iplus.reset(0)
    iplus.sxdg(0)

    return (
        InstanceGate(1, f"{ind}_0", [i, z, x, y]),
        InstanceGate(1, f"{ind}_1", [zero, one, plus, iplus]),
    )
