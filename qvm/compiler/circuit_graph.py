import itertools
from typing import NamedTuple

import networkx as nx
from qiskit.circuit import QuantumCircuit, Qubit

from qvm.tensor import HybridTensorNetwork, QuantumTensor, InstanceGate
from qvm.virtual_gates import VirtualBinaryGate, VIRTUAL_GATE_GENERATORS, VirtualMove


class CircuitGraphNode(NamedTuple):
    op_id: int
    qubit: Qubit


class CircuitGraph:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        graph = nx.MultiDiGraph()

        qubit_to_node = {}

        for op_id, instr in enumerate(circuit):
            if len(instr.qubits) > 2:
                raise ValueError(
                    f"Only 1 or 2 qubit gates are supported, got {instr.operation}"
                )

            # TODO check if gate is virtualizable

            for qubit in instr.qubits:
                new_node = CircuitGraphNode(op_id=op_id, qubit=qubit)
                graph.add_node(new_node)
                if qubit in qubit_to_node:
                    graph.add_edge(qubit_to_node[qubit], new_node, weight=4)

                qubit_to_node[qubit] = new_node

            if len(instr.qubits) == 2:
                graph.add_edge(
                    CircuitGraphNode(op_id=op_id, qubit=instr.qubits[0]),
                    CircuitGraphNode(op_id=op_id, qubit=instr.qubits[1]),
                    weight=5,
                )

        self._graph = graph

    def circuit(self) -> QuantumCircuit:
        return self._circuit.copy()

    def graph(self) -> nx.DiGraph:
         return self._graph.copy()


def index_str(u: CircuitGraphNode, v: CircuitGraphNode) -> str:
    return str(hash((u, v)))[:8]


# class SubCircuitGraph:
#     def __init__(
#         self, circuit_graph: CircuitGraph, node_subset: set[CircuitGraphNode]
#     ) -> None:
#         self._circuit_graph = circuit_graph
#         self._node_subset = node_subset

#     def graph(self) -> nx.DiGraph:
#         return self._circuit_graph.graph().subgraph(self._node_subset)

#     def to_quantum_tensor(self) -> QuantumTensor:
#         pass

    # def subcircuit(self) -> QuantumCircuit:
    #     circuit_graph = self._circuit_graph
    #     node_subset = self._node_subset

    #     circuit = circuit_graph.circuit()
    #     graph = circuit_graph.graph()

    #     subcircuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    #     virtual_gates: list[tuple[str, VirtualBinaryGate, int]] = []

    #     for op_id, instr in enumerate(circuit):
    #         op_nodes = {
    #             CircuitGraphNode(op_id=op_id, qubit=qubit) for qubit in instr.qubits
    #         }
    #         op_subset = op_nodes & node_subset
    #         # no operation on the subset
    #         if len(op_subset) == 0:
    #             continue

    #         op = instr.operation

    #         for op_node in op_subset:
    #             non_preds = set(graph.predecessors(op_node)) - node_subset

    #             for pred in non_preds:
    #                 if pred.op_id == op_node.op_id:
    #                     virtual_gates.append(
    #                         VirtualGateInfo(
    #                             index=index_str(pred, op_node),
    #                             vgate=VIRTUAL_GATE_GENERATORS[op.name](op.params),
    #                             qubit_id=1,
    #                         )
    #                     )
    #                 elif pred.qubit == op_node.qubit:
    #                     virtual_gates.append(
    #                         VirtualGateInfo(
    #                             index=index_str(pred, op_node),
    #                             vgate=VirtualMove(),
    #                             qubit_id=1,
    #                         )
    #                     )
    #                 else:
    #                     raise ValueError("Invalid edge", (pred, op_node))

    #                 subcircuit.append(
    #                     PlaceholderInstruction(index_str(pred, op_node)),
    #                     [op_node.qubit],
    #                 )

    #         if len(op_subset) == len(instr.qubits):
    #             subcircuit.append(instr)

    #         for op_node in op_subset:
    #             non_succs = set(graph.successors(op_node)) - node_subset

    #             for succ in non_succs:
    #                 if succ.op_id == op_node.op_id:
    #                     virtual_gates.append(
    #                         VirtualGateInfo(
    #                             index=index_str(op_node, succ),
    #                             vgate=VIRTUAL_GATE_GENERATORS[op.name](op.params),
    #                             qubit_id=0,
    #                         )
    #                     )
    #                 elif succ.qubit == op_node.qubit:
    #                     virtual_gates.append(
    #                         VirtualGateInfo(
    #                             index=index_str(op_node, succ),
    #                             vgate=VirtualMove(),
    #                             qubit_id=0,
    #                         )
    #                     )
    #                 else:
    #                     raise ValueError("Invalid edge", (op_node, succ))

    #                 subcircuit.append(
    #                     PlaceholderInstruction(index_str(op_node, succ)),
    #                     [op_node.qubit],
    #                 )

    #     return SubCircuit(subcircuit, virtual_gates)


def subgraph_to_quantum_tensor(
    graph: CircuitGraph,
    node_subset: set[CircuitGraphNode],
    remove_edges: set[tuple[CircuitGraphNode, CircuitGraphNode]],
) -> tuple[QuantumTensor, dict[str, VirtualBinaryGate]]:

    # TODO wire-cuts to moves
    pass


def circuit_graph_to_hybrid_tn(
    circuit_graph: CircuitGraph,
    connected_components: list[set[CircuitGraphNode]],
    remove_edges: set[tuple[CircuitGraphNode, CircuitGraphNode]],
) -> HybridTensorNetwork:
    graph = circuit_graph.graph()

    assert itertools.chain(*connected_components) == set(graph.nodes)


# if __name__ == "__main__":
#     qc = QuantumCircuit(2)
#     qc.h(0)
#     qc.cx(0, 1)
#     # qc.measure_all()

#     cg = CircuitGraph(qc)

#     from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection

#     A, B = kernighan_lin_bisection(cg.graph().to_undirected())

#     # print(cg.graph().edges)
#     print("A", A)
#     print("B", B)

#     x = subgraph_to_circuit_tensor(cg, A).circuit
#     y = subgraph_to_circuit_tensor(cg, B).circuit

#     print(x)
#     print(y)
