import itertools

import networkx as nx
from qiskit.circuit import (
    Qubit,
    QuantumCircuit,
    Barrier,
    QuantumRegister,
)

from qvm.virtual_gates import (
    VirtualBinaryGate,
    VirtualCX,
    VirtualCY,
    VirtualCZ,
    VirtualRZZ,
    VirtualIdentity,
)


VIRTUAL_GATE_TYPES: dict[str, type[VirtualBinaryGate]] = {
    "cx": VirtualCX,
    "cy": VirtualCY,
    "cz": VirtualCZ,
    "rzz": VirtualRZZ,
}


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


def connected_qubits(circuit: QuantumCircuit) -> list[set[Qubit]]:
    """Returns the connected qubits of a circuit.

    Args:
        circuit (QuantumCircuit): The circuit.

    Returns:
        list[set[Qubit]]: The connected qubits as a list of disjoint sets of qubits.
    """
    qcg = circuit_to_qcg(circuit)
    return list(nx.connected_components(qcg))


def fragment_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
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
    Decomposes a circuit using gate virtualization.
    The fragments are defined by the connected qubits, which should still be connected.

    Args:
        circuit (QuantumCircuit): The original circuit.
        con_qubits (list[set[Qubit]]): The connected qubits.
            Each set of qubits is a fragment.
            The qubit set need to be disjoint and contain all qubits of the circuit.

    Raises:
        ValueError: Thrown if con_qubits is illegal.

    Returns:
        QuantumCircuit: The decomposed circuit with virtual gates.
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
    return fragment_circuit(new_circ)


def remove_virtual_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    removed_circuit = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )
    for cinstr in circuit.data:
        if isinstance(cinstr.operation, VirtualBinaryGate):
            cinstr.operation = VirtualIdentity(cinstr.operation.original_gate)
        removed_circuit.append(cinstr.operation, cinstr.qubits, cinstr.clbits)
    return removed_circuit
