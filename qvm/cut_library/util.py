import itertools

import networkx as nx
from qiskit.circuit import (
    Qubit,
    QuantumCircuit,
    Barrier,
    QuantumRegister,
    Gate,
    Instruction,
)

from qvm.virtual_gates import (
    VirtualBinaryGate,
    VirtualQubitChannel,
    VirtualCX,
    VirtualCY,
    VirtualCZ,
    VirtualRZZ,
    WireCut,
)


VIRTUAL_GATE_TYPES: dict[str, type[VirtualBinaryGate]] = {
    "cx": VirtualCX,
    "cy": VirtualCY,
    "cz": VirtualCZ,
    "rzz": VirtualRZZ,
}


def append_virtual_gate(
    circuit: QuantumCircuit, original_gate: Gate, qubits: list[Qubit]
) -> None:
    """
    Appends a virtual gate to a circuit.

    Args:
        circuit (QuantumCircuit): The circuit.
        original_gate (Gate): The original gate of which a virtual gate should be inserted.
        qubits (list[Qubit]): The qubits on which the gate should be applied.
    """
    vgate = VIRTUAL_GATE_TYPES[original_gate.name](original_gate)
    circuit.append(vgate, qubits, [])


def circuit_to_qcg(circuit: QuantumCircuit) -> nx.Graph:
    """
    Transforms a circuit into a qubit connectivity graph (QCG).
    The QCG is a weighted graph, with the nodes being the qubits
    and the edges being the connections between the qubits.
    Each edge has a weight, which is the number of gates between the qubits.

    Args:
        circuit (QuantumCircuit): The circuit.

    Returns:
        nx.Graph: The quantum connectivity graph.
    """
    graph = nx.Graph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    graph.add_nodes_from(circuit.qubits)
    for cinstr in circuit.data:
        op, qubits = cinstr.operation, cinstr.qubits
        if isinstance(op, Barrier):
            continue
        if len(qubits) >= 2:
            for qubit1, qubit2 in itertools.combinations(qubits, 2):
                if not graph.has_edge(qubit1, qubit2):
                    graph.add_edge(qubit1, qubit2, weight=0)
                graph[qubit1][qubit2]["weight"] += 1
    return graph


def circuit_to_dag(circuit: QuantumCircuit) -> nx.DiGraph:
    """
    Converts a circuit into a simple directed acyclic graph (DAG).
    In this DAG, the nodes are the indices of the operations.

    Args:
        circuit (QuantumCircuit): The circuit.

    Returns:
        nx.DiGraph: The DAG.
    """

    graph = nx.DiGraph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    graph.add_nodes_from(range(len(circuit.data)))

    def _next_operation_on_qubit(from_index: int, qubit: Qubit) -> int:
        for i, cinstr in enumerate(circuit.data[from_index + 1 :]):
            if qubit in cinstr.qubits:
                return i + from_index + 1
        return -1

    for i, cinstr in enumerate(circuit.data):
        for qubit in cinstr.qubits:
            next_op = _next_operation_on_qubit(i, qubit)
            if next_op != -1:
                if not graph.has_edge(i, next_op):
                    graph.add_edge(i, next_op, weight=0)
                graph[i][next_op]["weight"] += 1

    return graph


class UnaryOperationSequence(Instruction):
    def __init__(self, operations: list[Instruction]):
        if not all(op.num_qubits == 1 for op in operations):
            raise ValueError("All operations must be unary.")
        super().__init__(name="seq", num_qubits=1, num_clbits=0, params=[])
        self._operations = operations

    def _define(self):
        circuit = QuantumCircuit(1, name=self.name)
        for op in self._operations:
            circuit.append(op, [0], [])
        self._definition = circuit


def compact_unary_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Compacts sequences of unary gates into single gates.
    This compaction can be reversed by calling `compacted_circuit.decompose()`.

    Args:
        circuit (QuantumCircuit): The circuit.

    Returns:
        QuantumCircuit: The compacted circuit.
    """

    compacted_circuit = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )

    def _next_unary_op(index: int, qubit: Qubit) -> tuple[int, Gate] | None:
        for i, cinstr in enumerate(circuit.data[index + 1 :]):
            if qubit in cinstr.qubits and (
                len(cinstr.qubits) > 1 or len(cinstr.clbits) > 0
            ):
                return None
            if qubit in cinstr.qubits and len(cinstr.qubits) == 1:
                return i + index + 1, cinstr.operation
        return None

    inserted: set[int] = set()
    for i, cinstr in enumerate(circuit.data):
        if len(cinstr.qubits) == 1 and len(cinstr.clbits) == 0:
            if i in inserted:
                continue
            ops = [cinstr.operation]
            j = i
            while (ind_op := _next_unary_op(j, cinstr.qubits[0])) is not None:
                j, op = ind_op
                if j not in inserted:
                    ops.append(op)
                inserted.add(j)

            compacted_circuit.append(UnaryOperationSequence(ops), cinstr.qubits, [])
        else:
            compacted_circuit.append(cinstr.operation, cinstr.qubits, cinstr.clbits)

    return compacted_circuit


def connected_qubits(circuit: QuantumCircuit) -> list[set[Qubit]]:
    """Returns the connected qubits of a circuit.

    Args:
        circuit (QuantumCircuit): The circuit.

    Returns:
        list[set[Qubit]]: The connected qubits as a list of disjoint sets of qubits.
    """
    qcg = circuit_to_qcg(circuit)
    return list(nx.connected_components(qcg))


def fragment_circuit(circuit: QuantumCircuit, vchan: bool = True) -> QuantumCircuit:
    """
    Creates a fragmented curcuit from a given circuit,
    where each fragment is a distinct quantum register.

    Args:
        circuit (QuantumCircuit): The original circuit.
        vchan (bool, optional): If virtual channels are considered. Defaults to True.

    Returns:
        QuantumCircuit: The fragmented circuit.
    """
    con_qubits = connected_qubits(circuit)
    new_frags = [
        QuantumRegister(len(qubits), name=f"frag{i}")
        for i, qubits in enumerate(con_qubits)
    ]
    qubit_map: dict[Qubit, Qubit] = {}  # old -> new Qubit
    for nodes, circ in zip(con_qubits, new_frags):
        node_l = list(nodes)
        for i in range(len(node_l)):
            qubit_map[node_l[i]] = circ[i]

    frag_circuit = QuantumCircuit(
        *new_frags,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )
    for circ_instr in circuit.data:
        frag_circuit.append(
            circ_instr.operation,
            [qubit_map[q] for q in circ_instr.qubits],
            circ_instr.clbits,
        )
    return frag_circuit


def decompose_qubits(
    circuit: QuantumCircuit, con_qubits: list[set[Qubit]]
) -> QuantumCircuit:
    """
    Decomposes a circuit into a fragemted circuit using gate virtualization,
    where each fragment is a distinct quantum register.
    The fragments are defined by the connected qubits, which should still be connected.

    Args:
        circuit (QuantumCircuit): The original circuit.
        con_qubits (list[set[Qubit]]): The connected qubits.
            Each set of qubits is a fragment.
            The qubit set need to be disjoint and contain all qubits of the circuit.

    Raises:
        ValueError: Thrown if con_qubits is illegal.

    Returns:
        QuantumCircuit: _description_
    """
    if set(circuit.qubits) != set.union(*con_qubits):
        raise ValueError("con_qubits is not containing all qubits of the circuit.")
    if len(list(itertools.chain(*con_qubits))) != len(circuit.qubits):
        raise ValueError("con_qubits is not disjoint.")

    def _in_multiple_fragments(qubits: set[Qubit]) -> bool:
        for qubit_set in con_qubits:
            if qubit_set & qubits and not qubits <= qubit_set:
                return True
            if qubits <= qubit_set:
                return False
        return False

    new_circ = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )
    for cinstr in circuit.data:
        op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
        if _in_multiple_fragments(set(qubits)) and not isinstance(op, Barrier):
            op = VIRTUAL_GATE_TYPES[op.name](op)
        new_circ.append(op, qubits, clbits)
    return fragment_circuit(new_circ, False)


class FadenBarrier(Barrier):
    """Barrier that holds two fragments together."""

    def __init__(self, orig_barrier: Barrier, orig_qubits: list[Qubit]):
        super().__init__(1, label="faden")
        self._orig_barrier = orig_barrier
        self._orig_qubits = orig_qubits


def extract_fragments(circuit: QuantumCircuit) -> dict[str, QuantumCircuit]:
    frag_circ = fragment_circuit(circuit)
    fragments: dict[str, QuantumCircuit] = {
        qreg.name: QuantumCircuit(qreg, *circuit.cregs) for qreg in frag_circ.qregs
    }

    def _append_faden_barrier(barrier: Barrier, qubits: list[Qubit]) -> None:
        for qubit in qubits:
            for circ in fragments.values():
                if qubit in circ.qregs[0]:
                    circ.append(FadenBarrier(barrier, qubits), [qubit], [])

    def _find_fragment(qubits: list[Qubit]) -> QuantumCircuit | None:
        for circ in fragments.values():
            if set(qubits) <= set(circ.qregs[0]):
                return circ
        return None

    for cinstr in frag_circ.data:
        frag = _find_fragment(cinstr.qubits)
        if isinstance(cinstr.operation, Barrier) and frag is None:
            _append_faden_barrier(cinstr.operation, cinstr.qubits)
        elif frag is not None:
            frag.append(cinstr.operation, cinstr.qubits, cinstr.clbits)
        else:
            raise ValueError("Could not find fragment for qubits.")
    return fragments


def stitch_fragments(fragments: dict[str, QuantumCircuit]) -> QuantumCircuit:
    pass


def remove_vswaps(circuit: QuantumCircuit) -> QuantumCircuit:
    pass


def wire_cuts_to_vswaps(circuit: QuantumCircuit) -> QuantumCircuit:
    pass
