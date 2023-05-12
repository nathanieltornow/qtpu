import itertools

import networkx as nx
from qiskit.circuit import Barrier, QuantumCircuit, QuantumRegister, Qubit

from qvm.types import Argument, PlaceholderGate
from qvm.virtual_gates import VIRTUAL_GATE_TYPES, VirtualSWAP, WireCut


def circuit_to_qcg(circuit: QuantumCircuit, use_qubit_idx: bool = False) -> nx.Graph:
    """
    Transforms a circuit into a qubit connectivity graph (QCG).
    The QCG is a weighted graph, with the nodes being the qubits
    and the edges being the connections between the qubits.
    Each edge has a weight, which is the number of gates between the qubits.

    Args:
        circuit (QuantumCircuit): The circuit.
        use_qubit_idx (bool): If the indexes of the qubits should be used
            instead of the qubit objects.

    Returns:
        nx.Graph: The quantum connectivity graph.
    """
    graph = nx.Graph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    if use_qubit_idx:
        graph.add_nodes_from(range(circuit.num_qubits))
    else:
        graph.add_nodes_from(circuit.qubits)

    for cinstr in circuit.data:
        op, qubits = cinstr.operation, cinstr.qubits
        if isinstance(op, Barrier):
            continue
        if len(qubits) >= 2:
            for qubit1, qubit2 in itertools.combinations(qubits, 2):
                if use_qubit_idx:
                    qubit1, qubit2 = circuit.qubits.index(qubit1), circuit.qubits.index(
                        qubit2
                    )

                if not graph.has_edge(qubit1, qubit2):
                    graph.add_edge(qubit1, qubit2, weight=0)
                graph[qubit1][qubit2]["weight"] += 1
    return graph


def fold_circuit(
    circuit: QuantumCircuit,
) -> tuple[QuantumCircuit, list[QuantumCircuit]]:
    _2q_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    _1q_circs: list[QuantumCircuit] = []
    current_1q_circ = QuantumCircuit(*circuit.qregs, *circuit.cregs, name="1q")

    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        if len(qubits) == 1 or isinstance(op, Barrier):
            current_1q_circ.append(op, qubits, clbits)
        elif len(qubits) == 2:
            _1q_circs.append(current_1q_circ)
            current_1q_circ = QuantumCircuit(*circuit.qregs, *circuit.cregs, name="1q")
            _2q_circuit.append(op, qubits, clbits)
        else:
            raise Exception("?")
    _1q_circs.append(current_1q_circ)
    return _2q_circuit, _1q_circs


def unfold_circuit(
    two_qubit_circuit: QuantumCircuit, one_qubit_circuits: list[QuantumCircuit]
) -> QuantumCircuit:
    if len(two_qubit_circuit) == 0:
        raise Exception("?")
    res_circuit = QuantumCircuit(*two_qubit_circuit.qregs, *two_qubit_circuit.cregs)
    res_circuit = res_circuit.compose(one_qubit_circuits[0])
    for instr2q, circ1q in zip(two_qubit_circuit, one_qubit_circuits[1:]):
        res_circuit.append(instr2q)
        res_circuit = res_circuit.compose(circ1q)
    return res_circuit


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
    con_qubits = list(nx.connected_components(circuit_to_qcg(circuit)))
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


def insert_placeholders(
    circuit: QuantumCircuit,
    arg: Argument,
) -> QuantumCircuit:
    new_circuit = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )
    for cinstr in circuit.data:
        op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
        if isinstance(op, PlaceholderGate):
            if op.key not in arg:
                raise ValueError(f"Missing insertion for placeholder {op.key}")
            if op.clbit is not None:
                new_circuit.compose(
                    arg[op.key], qubits, clbits=[op.clbit], inplace=True
                )
            else:
                new_circuit.compose(arg[op.key], qubits, inplace=True)
        else:
            new_circuit.append(op, qubits, clbits)
    return new_circuit


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


def wirecuts_to_vswaps(circuit: QuantumCircuit) -> QuantumCircuit:
    if sum(1 for instr in circuit if isinstance(instr, VirtualSWAP)) > 0:
        raise ValueError("Circuit already contains virtual SWAP gates.")
    num_wire_cuts = sum(1 for instr in circuit if isinstance(instr.operation, WireCut))
    if num_wire_cuts == 0:
        return circuit.copy()

    wire_cut_register = QuantumRegister(num_wire_cuts, "wire_cut")

    new_circuit = QuantumCircuit(
        *circuit.qregs,
        wire_cut_register,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )
    qubit_map: dict[Qubit, Qubit] = {}

    def _find_qubit(qubit: Qubit) -> Qubit:
        while qubit in qubit_map:
            qubit = qubit_map[qubit]
        return qubit

    cut_ctr = 0
    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        qubits = [_find_qubit(qubit) for qubit in qubits]
        if isinstance(op, WireCut):
            qubit_map[qubits[0]] = wire_cut_register[cut_ctr]
            op = VirtualSWAP()
            qubits = [qubits[0], wire_cut_register[cut_ctr]]
            cut_ctr += 1
        new_circuit.append(op, qubits, clbits)
    return new_circuit


def virtualize_between_qubits(
    circuit: QuantumCircuit, qubit1: Qubit, qubit2: Qubit, max_virtual_gates: int = 4
) -> tuple[QuantumCircuit, int]:
    """_summary_

    Args:
        circuit (QuantumCircuit): _description_
        qubit1 (Qubit): _description_
        qubit2 (Qubit): _description_
        max_virtual_gates (int, optional): _description_. Defaults to 4.

    Raises:
        ValueError: If the number of virtual gates that need to be inserted exceeds the maximum.

    Returns:
        tuple[QuantumCircuit, int]: _description_
    """
    new_circuit = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )
    num_virtual_gates = 0
    for cinstr in circuit.data:
        op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
        if qubit1 in qubits and qubit2 in qubits and not isinstance(op, Barrier):
            op = VIRTUAL_GATE_TYPES[op.name](op)
            num_virtual_gates += 1
            if num_virtual_gates > max_virtual_gates:
                raise ValueError(
                    f"Number of virtual gates exceeds the maximum of {max_virtual_gates}."
                )
        new_circuit.append(op, qubits, clbits)
    return new_circuit, num_virtual_gates
