import abc

import networkx as nx
from qiskit.circuit import Barrier, QuantumCircuit, QuantumRegister, Qubit

from qvm.instructions import VirtualBinaryGate, WireCut
from qvm.virtual_gates import VIRTUAL_GATE_GENERATORS, VirtualMove
from ._graphs import TNGraph, QubitGraph


class Cutter(abc.ABC):
    @abc.abstractmethod
    def _cut(self, tn_graph: TNGraph) -> set[tuple[int, int]]:
        pass

    def _cut_cost(self, tn_graph: TNGraph, cut_edges: set[tuple[int, int]]) -> int:
        g = tn_graph.copy()
        print(type(g))
        g.remove_edges_from(cut_edges)
        ccs = list(nx.connected_components(g))

        def _find_components(u: int, v: int) -> tuple[int, int]:
            u_cc, v_cc = -1, -1
            for i, cc in enumerate(ccs):
                if u in cc:
                    u_cc = i
                if v in cc:
                    v_cc = i
            if u_cc == -1 or v_cc == -1:
                raise ValueError("Invalid edge")
            return u_cc, v_cc

        costs = [1] * len(ccs)

        for u, v in cut_edges:
            u_cc, v_cc = _find_components(u, v)
            if u_cc == v_cc:
                cut_edges.remove((u, v))
                continue

            # u_cc != v_cc
            if tn_graph.is_wire_edge(u, v):
                costs[u_cc] *= 4
                costs[v_cc] *= 4
            elif tn_graph.is_gate_edge(u, v):
                costs[u_cc] *= 5
                costs[v_cc] *= 5
            else:
                raise ValueError("Invalid edge")

        return sum(costs)

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        tn_graph = TNGraph(circuit)
        cut_edges = self._cut(tn_graph)

        vgates: dict[int, VirtualBinaryGate] = {}
        wire_cuts: dict[int, int] = {}
        for node1, node2 in cut_edges:
            assert tn_graph.has_edge(node1, node2)

            instr1, instr2 = (
                tn_graph.nodes[node1]["instr_idx"],
                tn_graph.nodes[node2]["instr_idx"],
            )
            qubit1, qubit2 = (
                tn_graph.nodes[node1]["qubit_idx"],
                tn_graph.nodes[node2]["qubit_idx"],
            )

            if instr1 == instr2:
                old_op = circuit[instr1].operation
                if isinstance(old_op, VirtualBinaryGate) or isinstance(old_op, Barrier):
                    continue
                vgate = VIRTUAL_GATE_GENERATORS[old_op.name](old_op.params)
                vgates[instr1] = vgate

            elif qubit1 == qubit2:
                first_instr = min(instr1, instr2)
                wire_cuts[first_instr] = tn_graph.nodes[node1]["qubit_idx"]

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

        res_circuit = wire_cuts_to_moves(res_circuit)
        res_circuit = decompose_circuit(res_circuit)
        return res_circuit


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
    qubit_graph = QubitGraph(circuit)
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
