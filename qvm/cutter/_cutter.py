import abc

import networkx as nx
from qiskit.circuit import Barrier, QuantumCircuit, QuantumRegister, Qubit

from qvm.instructions import VirtualBinaryGate, WireCut
from qvm.virtual_gates import VIRTUAL_GATE_GENERATORS, VirtualMove


class Cutter(abc.ABC):
    @abc.abstractmethod
    def _run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit = self._run(circuit)
        circuit = wire_cuts_to_moves(circuit)
        circuit = decompose_circuit(circuit)
        return circuit


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


class TNCutter(Cutter, abc.ABC):
    @abc.abstractmethod
    def _cut_tn(self, tn_graph: nx.Graph) -> list[tuple[int, int]]:
        pass

    def _run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit = circuit.copy()
        cut_graph = circuit_to_tn_graph(circuit)
        cut_edges = self._cut_tn(cut_graph)

        vgates: dict[int, VirtualBinaryGate] = {}
        wire_cuts: dict[int, int] = {}
        for node1, node2 in cut_edges:
            assert cut_graph.has_edge(node1, node2)

            instr1, qubit1 = (
                cut_graph.nodes[node1]["instr_idx"],
                cut_graph.nodes[node1]["qubit_idx"],
            )
            instr2, qubit2 = (
                cut_graph.nodes[node2]["instr_idx"],
                cut_graph.nodes[node2]["qubit_idx"],
            )

            if instr1 == instr2:
                old_op = circuit[instr1].operation
                if isinstance(old_op, VirtualBinaryGate) or isinstance(old_op, Barrier):
                    continue
                vgate = VIRTUAL_GATE_GENERATORS[old_op.name](old_op.params)
                vgates[instr1] = vgate

            elif qubit1 == qubit2:
                first_instr = min(instr1, instr2)
                wire_cuts[first_instr] = cut_graph.nodes[node1]["qubit_idx"]

            else:
                raise ValueError("Invalid cut")

        res_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

        for i, instr in enumerate(circuit):
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
            if i in vgates:
                op = vgates[i]

            res_circuit.append(op, qubits, clbits)

            if i in wire_cuts:
                res_circuit.append(WireCut(), [wire_cuts[i]], [])

        return res_circuit


class QubitGraphCutter(Cutter, abc.ABC):
    @abc.abstractmethod
    def _cut_qubit_graph(self, qubit_graph: nx.Graph) -> list[tuple[int, int]]:
        pass

    def _run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        qubit_graph = circuit_to_qubit_graph(circuit)
        cut_edges = self._cut_qubit_graph(qubit_graph)

        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        for instr in circuit:
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits

            if len(qubits) == 2:
                q1_idx, q2_idx = [circuit.qubits.index(qubit) for qubit in qubits]
                if (q1_idx, q2_idx) in cut_edges or (q2_idx, q1_idx) in cut_edges:
                    op = VIRTUAL_GATE_GENERATORS[op.name](op.params)

            new_circuit.append(op, qubits, clbits)

        return new_circuit


class PortGraphCutter(Cutter, abc.ABC):
    @abc.abstractmethod
    def _cut_portgraph(self, port_graph: nx.DiGraph) -> list[tuple[int, int]]:
        ...

    def _run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        port_graph = circuit_to_portgraph(circuit)
        cut_edges = self._cut_portgraph(port_graph)

        # instr_idx -> port
        wire_cuts: dict[int, int] = {}

        for cu, cv in cut_edges:
            assert port_graph.has_edge(cu, cv)

            instr_idx = cu
            port_idx = port_graph[cu][cv]["from_qubit"]

            wire_cuts[instr_idx] = port_idx

        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

        for instr_idx, instr in enumerate(circuit):
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits

            new_circuit.append(op, qubits, clbits)
            if instr_idx in wire_cuts:
                new_circuit.append(WireCut(), [qubits[wire_cuts[instr_idx]]], [])

        return new_circuit


def circuit_to_portgraph(circuit: QuantumCircuit) -> nx.DiGraph:
    graph = nx.DiGraph()

    current_nodes: dict[int, tuple[int, int]] = {
        qubit: (-1, -1) for qubit in circuit.qubits
    }

    for instr_idx, instr in enumerate(circuit):
        graph.add_node(instr_idx)

        for qubit_idx, qubit in enumerate(instr.qubits):
            if current_nodes[qubit] != (-1, -1):
                prev_node, prev_qubit_idx = current_nodes[qubit]
                graph.add_edge(
                    prev_node,
                    instr_idx,
                    from_qubit=prev_qubit_idx,
                    to_qubit=qubit_idx,
                )

            current_nodes[qubit] = (instr_idx, qubit_idx)

    return graph


def circuit_to_qubit_graph(circuit: QuantumCircuit) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(circuit.qubits)))

    for instr in circuit:
        if (
            len(instr.qubits) == 1
            or isinstance(instr.operation, Barrier)
            or isinstance(instr.operation, VirtualBinaryGate)
        ):
            continue
        for qubit1, qubit2 in zip(instr.qubits, instr.qubits[1:]):
            q1_idx, q2_idx = circuit.qubits.index(qubit1), circuit.qubits.index(qubit2)
            if graph.has_edge(q1_idx, q2_idx):
                graph[q1_idx][q2_idx]["weight"] += 1
            graph.add_edge(
                circuit.qubits.index(qubit1), circuit.qubits.index(qubit2), weight=1
            )

    return graph


def circuit_to_tn_graph(
    circuit: QuantumCircuit, wire_cost: int = 4, gate_cost: int = 5
) -> nx.Graph:
    graph = nx.Graph()

    # qubit_idx -> node_idx
    current_nodes: dict[int, int] = {i: -1 for i in range(len(circuit.qubits))}

    nodeidx = 0
    for instr_idx, instr in enumerate(circuit):
        added_nodes: list[int] = []

        for qubit in instr.qubits:
            qubit_idx = circuit.qubits.index(qubit)

            graph.add_node(nodeidx, qubit_idx=qubit_idx, instr_idx=instr_idx)
            added_nodes.append(nodeidx)

            if current_nodes[qubit_idx] != -1:
                graph.add_edge(current_nodes[qubit_idx], nodeidx, weight=wire_cost)
            current_nodes[qubit_idx] = nodeidx

            nodeidx += 1

        if isinstance(instr.operation, VirtualBinaryGate | Barrier):
            continue

        for i in range(len(added_nodes) - 1):
            graph.add_edge(added_nodes[i], added_nodes[i + 1], weight=gate_cost)

    return graph
