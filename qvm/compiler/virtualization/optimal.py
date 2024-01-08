import abc

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import QuantumCircuit

from qvm.compiler.types import VirtualizationPass

from qvm.virtual_gates import VIRTUAL_GATE_TYPES, WireCut, VirtualBinaryGate


class Cutter(abc.ABC):
    @abc.abstractmethod
    def _cut(self, cut_graph: nx.Graph) -> list[tuple[int, int]]:
        pass

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit = circuit.copy()
        cut_graph = circuit_to_cutgraph(circuit)
        cut_edges = self._cut(cut_graph)

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
                vgate = VIRTUAL_GATE_TYPES[old_op.name](old_op)
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


class BisectionCutter(Cutter):
    def _cut(self, cut_graph: nx.Graph) -> list[tuple[int, int]]:
        A, B = kernighan_lin_bisection(cut_graph)
        cut_edges = []
        for node1, node2 in cut_graph.edges:
            if (node1 in A and node2 in B) or (node1 in B and node2 in A):
                cut_edges.append((node1, node2))
        return cut_edges


def circuit_to_cutgraph(circuit: QuantumCircuit) -> nx.Graph:
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
                graph.add_edge(current_nodes[qubit_idx], nodeidx)
            current_nodes[qubit_idx] = nodeidx

            nodeidx += 1

        for i in range(len(added_nodes) - 1):
            graph.add_edge(added_nodes[i], added_nodes[i + 1])

    return graph


def cut_graph_to_asp(cut_graph: nx.Graph) -> str:
    asp = ""
    for node, qubit_idx in cut_graph.nodes.data("qubit_idx"):
        asp += f"node({node}, {qubit_idx}).\n"
    for node1, node2 in cut_graph.edges:
        asp += f"edge({node1}, {node2}).\n"
    return asp


if __name__ == "__main__":
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    circuit.cx(1, 2)

    cutter = BisectionCutter()
    c = cutter.run(circuit)
    print(c)
