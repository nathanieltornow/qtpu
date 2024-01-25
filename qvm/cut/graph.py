import networkx as nx
from qiskit.circuit import QuantumCircuit, Barrier, QuantumRegister, Qubit

from qvm.instructions import VirtualBinaryGate, WireCut
from qvm.virtual_gates import VIRTUAL_GATE_GENERATORS, VirtualMove


class CircuitGraph:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._graph = circuit_to_graph(circuit)

    def get_nx_graph(self) -> nx.Graph:
        return self._graph.copy()

    def nodes(self) -> list[int]:
        return list(self._graph.nodes())

    def node_info(self, node: int) -> dict[str, int]:
        self.subgraph()
        return self._graph.nodes[node]

    def generate_circuit(self, removed_edges: set[tuple[int, int]]) -> QuantumCircuit:
        vgates: dict[int, VirtualBinaryGate] = {}
        wire_cuts: dict[int, set[int]] = {}

        for node1, node2 in removed_edges:
            # assert self._graph.has_edge(node1, node2)

            instr1, instr2 = (
                self._graph.nodes[node1]["instr_idx"],
                self._graph.nodes[node2]["instr_idx"],
            )
            qubit1, qubit2 = (
                self._graph.nodes[node1]["qubit_idx"],
                self._graph.nodes[node2]["qubit_idx"],
            )

            if instr1 == instr2:
                old_op = self._circuit[instr1].operation
                if isinstance(old_op, VirtualBinaryGate) or isinstance(old_op, Barrier):
                    continue
                vgate = VIRTUAL_GATE_GENERATORS[old_op.name](old_op.params)
                vgates[instr1] = vgate

            elif qubit1 == qubit2:
                first_instr = min(instr1, instr2)
                if first_instr not in wire_cuts:
                    wire_cuts[first_instr] = set()
                wire_cuts[first_instr].add(self._graph.nodes[node1]["qubit_idx"])

            else:
                raise ValueError("Invalid cut")

        res_circuit = QuantumCircuit(*self._circuit.qregs, *self._circuit.cregs)

        for i, instr in enumerate(self._circuit):
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
            if i in vgates:
                op = vgates[i]

            res_circuit.append(op, qubits, clbits)

            if i in wire_cuts:
                for qubit_idx in wire_cuts[i]:
                    res_circuit.append(WireCut(), [qubit_idx], [])

        res_circuit = wire_cuts_to_moves(res_circuit)
        res_circuit = decompose_circuit(res_circuit)
        return res_circuit


def circuit_to_graph(
    circuit: QuantumCircuit, wire_cost: int = 4, gate_cost: int = 5
) -> nx.Graph:
    graph = nx.Graph()

    current_nodes: dict[int, int] = {i: -1 for i in range(len(circuit.qubits))}

    nodeidx = 0
    for instr_idx, instr in enumerate(circuit):
        added_nodes: list[int] = []

        if len(instr.qubits) == 1 or isinstance(
            instr.operation, VirtualBinaryGate | Barrier
        ):
            continue

        for qubit in instr.qubits:
            qubit_idx = circuit.qubits.index(qubit)

            graph.add_node(nodeidx, qubit_idx=qubit_idx, instr_idx=instr_idx)
            added_nodes.append(nodeidx)

            if current_nodes[qubit_idx] != -1:
                graph.add_edge(current_nodes[qubit_idx], nodeidx, weight=wire_cost)
            current_nodes[qubit_idx] = nodeidx

            nodeidx += 1

        for i in range(len(added_nodes) - 1):
            graph.add_edge(added_nodes[i], added_nodes[i + 1], weight=gate_cost)
    return graph


def decompose_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    qubit_graph = circuit_to_qubit_graph(circuit)
    qubit_mapping: dict[Qubit, Qubit] = {}
    new_qregs = []
    frag_ctr = 0
    for q_idx_set in nx.connected_components(qubit_graph):
        q_ids = sorted(q_idx_set)
        qreg = QuantumRegister(len(q_ids), f"frag{frag_ctr}")
        new_qregs.append(qreg)
        old_qubits = [circuit.qubits[q_id] for q_id in q_ids]
        qubit_mapping |= dict(zip(old_qubits, qreg))
        frag_ctr += 1

    new_circuit = QuantumCircuit(*new_qregs, *circuit.cregs)
    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        qubits = [qubit_mapping[qubit] for qubit in qubits]
        new_circuit.append(op, qubits, clbits)

    return new_circuit


def circuit_to_qubit_graph(circuit: QuantumCircuit) -> nx.Graph:
    qubit_graph = nx.Graph()
    qubit_graph.add_nodes_from(range(len(circuit.qubits)))

    for instr in circuit:
        if (
            len(instr.qubits) == 1
            or isinstance(instr.operation, Barrier)
            or isinstance(instr.operation, VirtualBinaryGate)
        ):
            continue
        for qubit1, qubit2 in zip(instr.qubits, instr.qubits[1:]):
            q1_idx, q2_idx = (
                circuit.qubits.index(qubit1),
                circuit.qubits.index(qubit2),
            )
            if qubit_graph.has_edge(q1_idx, q2_idx):
                qubit_graph[q1_idx][q2_idx]["weight"] += 1
            qubit_graph.add_edge(
                circuit.qubits.index(qubit1), circuit.qubits.index(qubit2), weight=1
            )
    return qubit_graph


def wire_cuts_to_moves(circuit: QuantumCircuit) -> QuantumCircuit:
    qubit_mapping: dict[Qubit, Qubit] = {}

    def _find_qubit(qubit: Qubit) -> Qubit:
        while qubit in qubit_mapping:
            qubit = qubit_mapping[qubit]
        return qubit

    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    move_reg = QuantumRegister(
        sum(1 for instr in circuit if isinstance(instr.operation, WireCut)), "vmove"
    )
    new_circuit.add_register(move_reg)

    cut_ctr = 0
    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        qubits = [_find_qubit(qubit) for qubit in qubits]
        if isinstance(op, WireCut):
            qubit_mapping[qubits[0]] = move_reg[cut_ctr]
            qubits = [qubits[0], move_reg[cut_ctr]]
            op = VirtualMove()
            cut_ctr += 1

        new_circuit.append(op, qubits, clbits)

    return new_circuit
