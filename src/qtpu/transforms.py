"""Functions for transforming quantum circuits into hybrid tensor networks."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import (
    CircuitInstruction,
    ClassicalRegister,
    Clbit,
    Instruction,
    Measure,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import (
    HGate,
    Reset,
    SXdgGate,
    SXGate,
    XGate,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_addon_cutting.instructions import CutWire, Move
from qiskit_addon_cutting.qpd import QPDMeasure, TwoQubitQPDGate

from qtpu.tensor import CircuitTensor, HybridTensorNetwork, InstructionVector


def insert_cuts(
    circuit: QuantumCircuit,
    gate_cuts: set[int],
    wire_cuts: set[tuple[int, int]],
    inplace: bool = False,
) -> QuantumCircuit:
    """Inserts gate and wire cuts into a quantum circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to modify.
        gate_cuts (set[int]): A set of indices where gate cuts should be inserted.
        wire_cuts (set[tuple[int, int]]): A set of tuples, each containing an operation index and
            a relative qubit index, after which a wire cut should be inserted.
        inplace (bool, optional): If True, modifies the circuit in place. If False,
            creates a copy of the circuit and modifies the copy. Defaults to False.

    Returns:
        QuantumCircuit: The modified quantum circuit with the specified cuts inserted.
    """
    if not inplace:
        circuit = circuit.copy()

    wire_cut_dict = dict(wire_cuts)

    indices = sorted(set(gate_cuts) | set(wire_cut_dict.keys()), reverse=True)

    for i in indices:
        gate = circuit.data[i]
        if i in gate_cuts:
            circuit.data[i] = CircuitInstruction(
                operation=TwoQubitQPDGate.from_instruction(gate.operation),
                qubits=gate.qubits,
                clbits=gate.clbits,
            )

        if i in wire_cut_dict:
            circuit.data.insert(
                i + 1, CircuitInstruction(CutWire(), [gate.qubits[0]], [])
            )

    return circuit


def wire_cuts_to_moves(
    circuit: QuantumCircuit, inplace: bool = False
) -> QuantumCircuit:
    """Replaces CutWire operations with virtual Move operations.

    Args:
        circuit (QuantumCircuit): The quantum circuit to transform.
        inplace (bool): If True, modifies the circuit in place. If False,
            creates a copy of the circuit and modifies the copy. Default is False.

    Returns:
        QuantumCircuit: The transformed quantum circuit with CutWire operations replaced by Move operations.
    """
    if not inplace:
        circuit = circuit.copy()

    num_wire_cuts = sum(1 for instr in circuit if isinstance(instr.operation, CutWire))

    qubit_mapping = {}

    def _find_qubit(qubit: Qubit) -> Qubit:
        while qubit in qubit_mapping:
            qubit = qubit_mapping[qubit]
        return qubit

    wire_cut_reg = QuantumRegister(num_wire_cuts, name="wire_cut")
    circuit.add_register(wire_cut_reg)

    ctr = 0
    for i in range(len(circuit.data)):
        gate = circuit.data[i]
        op, qubits = gate.operation, gate.qubits

        qubits = [_find_qubit(qubit) for qubit in qubits]

        if isinstance(op, CutWire):
            qubit_mapping[qubits[0]] = wire_cut_reg[ctr]
            qubits += [wire_cut_reg[ctr]]
            ctr += 1
            op = TwoQubitQPDGate.from_instruction(Move())

        circuit.data[i] = CircuitInstruction(op, qubits, gate.clbits)
    return circuit


def circuit_to_hybrid_tn(circuit: QuantumCircuit) -> HybridTensorNetwork:
    """Convert a quantum circuit to a hybrid tensor network.

    Args:
        circuit (QuantumCircuit): The quantum circuit to be converted.

    Returns:
        HybridTensorNetwork: The resulting hybrid tensor network.
    """
    circuit = seperate_clbits(circuit)

    graph = _qubit_graph(circuit)
    ccs = list(nx.connected_components(graph))
    qubit_components = {qubit: i for i, cc in enumerate(ccs) for qubit in cc}

    qpd_inds = [
        i
        for i, instr in enumerate(circuit.data)
        if isinstance(instr.operation, TwoQubitQPDGate)
    ]

    ctensors = []

    for i, ind in enumerate(reversed(qpd_inds)):
        gate = circuit.data[ind]

        idx = f"i_{len(qpd_inds) - i - 1}"

        param1 = param2 = Parameter(idx)
        op_vector1, op_vector2 = zip(*gate.operation.basis.maps, strict=False)

        ct = qtn.Tensor(gate.operation.basis.coeffs, inds=[idx], tags=["QPD"])

        if (
            qubit_components is not None
            and qubit_components[gate.qubits[0]] != qubit_components[gate.qubits[1]]
            and gate.label == "cut_move"
        ):
            # override with CutQC-style wire cut if the qubits are in different components
            wire_matrix, op_vector1, op_vector2 = _generate_wire_data()
            param1 = Parameter(idx + "_0")
            param2 = Parameter(idx + "_1")
            ct = qtn.Tensor(wire_matrix, inds=[idx + "_0", idx + "_1"], tags=["wire"])

        circuit.data[ind] = CircuitInstruction(
            operation=InstructionVector(op_vector1, param1),
            qubits=[gate.qubits[0]],
            clbits=[],
        )
        circuit.data.insert(
            ind + 1,
            CircuitInstruction(
                operation=InstructionVector(op_vector2, param2),
                qubits=[gate.qubits[1]],
                clbits=[],
            ),
        )
        ctensors.append(ct)

    subcircuits = [circuit_on_component(circuit, cc) for cc in ccs]
    subcircuits = [remove_idle_cregs(sc) for sc in subcircuits]
    qtensors = [CircuitTensor(subcircuit) for subcircuit in subcircuits]

    return HybridTensorNetwork(qtensors, ctensors)


def circuit_on_component(circuit: QuantumCircuit, qubits: set[Qubit]) -> QuantumCircuit:
    """Extracts a subcircuit from the given quantum circuit that operates only on the specified qubits.

    Args:
        circuit (QuantumCircuit): The original quantum circuit.
        qubits (set[Qubit]): A set of qubits to extract the subcircuit for.

    Returns:
        QuantumCircuit: A new quantum circuit that contains only the operations from the original circuit
                        that act exclusively on the specified qubits.

    Raises:
        ValueError: If any gate in the original circuit acts on qubits in *and* out of the specified set.
    """
    qreg = QuantumRegister(len(qubits), "q")
    qubit_mapping = {
        qubit: qreg[i]
        for i, qubit in enumerate(sorted(qubits, key=circuit.qubits.index))
    }
    new_circuit = QuantumCircuit(qreg, *circuit.cregs)
    for instr in circuit:
        if set(instr.qubits) <= qubits:
            new_circuit.append(
                instr.operation,
                [qubit_mapping[qubit] for qubit in instr.qubits],
                instr.clbits,
            )

        elif set(instr.qubits) & qubits:
            msg = "Gate acts on qubits not in the component."
            raise ValueError(msg)

    return new_circuit


def decompose_qpd_measures(
    circuit: QuantumCircuit, defer: bool = True, inplace: bool = True
) -> QuantumCircuit:
    """Decomposes QPD (Quantum Phase Detection) measure operations in a given quantum circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit containing QPD measure operations.
        defer (bool, optional): If True, defers the measurement using an auxiliary qubit. Defaults to True.
        inplace (bool, optional): If True, modifies the circuit in place. If False,
            creates a copy of the circuit. Defaults to True.

    Returns:
        QuantumCircuit: The modified quantum circuit with QPD measure operations decomposed.
    """
    if not inplace:
        circuit = circuit.copy()

    qpd_measure_ids = [
        i
        for i, instruction in enumerate(circuit.data)
        if instruction.operation.name.lower() == "qpd_measure"
    ]
    if len(qpd_measure_ids) == 0:
        return circuit

    reg = ClassicalRegister(len(qpd_measure_ids), name="qpd_measurements")
    circuit.add_register(reg)

    qreg = QuantumRegister(len(qpd_measure_ids), name="deffered_qubits")

    if defer:
        circuit.add_register(qreg)

        def _defer_circuit() -> QuantumCircuit:
            defer_circuit = QuantumCircuit(2, 1)
            defer_circuit.cx(0, 1)
            defer_circuit.measure(1, 0)
            return defer_circuit

    for idx, i in enumerate(qpd_measure_ids):
        gate = circuit.data[i]

        inst = CircuitInstruction(
            operation=Measure(), qubits=[gate.qubits], clbits=[reg[idx]]
        )

        if defer:
            inst = CircuitInstruction(
                operation=_defer_circuit().to_instruction(),
                qubits=[gate.qubits, qreg[idx]],
                clbits=[reg[idx]],
            )

        circuit.data[i] = inst

    return circuit


def remove_operations_by_name(
    circuit: QuantumCircuit, names: str | set[str], inplace: bool = True
) -> QuantumCircuit:
    """Remove operations from a QuantumCircuit by their names.

    Args:
        circuit (QuantumCircuit): The quantum circuit from which to remove operations.
        names (str or set[str]): The name or set of names of the operations to remove.
        inplace (bool): If True, modify the circuit in place. If False,
            return a copy of the circuit with the operations removed. Default is True.

    Returns:
        QuantumCircuit: The modified quantum circuit with the specified operations removed.
    """
    if not inplace:
        circuit = circuit.copy()

    if isinstance(names, str):
        names = {names}

    measure_ids_qubits = [
        i
        for i, instruction in enumerate(circuit.data)
        if instruction.operation.name in names
    ]

    for i in reversed(measure_ids_qubits):
        circuit.data.pop(i)

    return circuit


def seperate_clbits(circuit: QuantumCircuit) -> QuantumCircuit:
    """Create a new QuantumCircuit with classical bits each placed in their own ClassicalRegister.

    Args:
        circuit (QuantumCircuit): The input quantum circuit to be transformed.

    Returns:
        QuantumCircuit: A new quantum circuit with the same classical bits
        and operations, but with classical bits each placed in their own
        ClassicalRegister.
    """
    cregs = [ClassicalRegister(1, f"c{i}") for i in range(circuit.num_clbits)]
    clbit_to_creg = dict(zip(circuit.clbits, cregs, strict=False))

    new_circuit = QuantumCircuit(*circuit.qregs, *cregs)
    for instr in circuit:
        new_circuit.append(
            instr.operation,
            instr.qubits,
            [clbit_to_creg[clbit][0] for clbit in instr.clbits],
        )
    return new_circuit


def remove_idle_cregs(circuit: QuantumCircuit) -> QuantumCircuit:
    """Removes classical registers that have at least one unused bit.

    Args:
        circuit (QuantumCircuit): The input circuit.

    Returns:
        QuantumCircuit: The circuit with classical registers having at
            least one unused classical bit removed.
    """
    dag = circuit_to_dag(circuit)
    idle_bits = list(dag.idle_wires())
    for bit in idle_bits:
        if isinstance(bit, Clbit):
            dag.remove_clbits(bit)
    return dag_to_circuit(dag)


def _qubit_graph(circuit: QuantumCircuit) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(circuit.qubits)
    for instr in circuit:
        if instr.operation.name == "barrier" or isinstance(
            instr.operation, TwoQubitQPDGate
        ):
            continue

        qubits = instr.qubits
        for qubit1, qubit2 in combinations(qubits, 2):
            graph.add_edge(qubit1, qubit2)
    return graph


def _generate_wire_data() -> (
    tuple[np.ndarray, list[list[Instruction]], list[list[Instruction]]]
):
    op_vector1 = [
        [Reset()],
        [QPDMeasure(), Reset()],
        [HGate(), QPDMeasure(), Reset()],
        [SXGate(), QPDMeasure(), Reset()],
    ]
    op_vector2 = [
        [Reset()],
        [Reset(), XGate()],
        [Reset(), HGate()],
        [Reset(), SXdgGate()],
    ]
    a = np.array(
        [[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    b = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [-1, -1, 2, 0], [-1, -1, 0, 2]],
        dtype=np.float32,
    )
    return 0.5 * a @ b, op_vector1, op_vector2
