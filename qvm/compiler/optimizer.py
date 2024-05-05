import abc
from typing import Callable
from itertools import chain

import networkx as nx

from qvm.circuit_graph import CircuitGraph, CircuitGraphNode
from qvm.compiler.knit_tree import KnitTree


class KnitTreeOptimizer(abc.ABC):
    @abc.abstractmethod
    def optimize(self, knit_tree: KnitTree) -> None: ...


class BisectionOptimizer(KnitTreeOptimizer, abc.ABC):
    @abc.abstractmethod
    def _next_leaf(self, node: KnitTree) -> KnitTree | None: ...

    def optimize(self, knit_tree: KnitTree) -> None:
        while (leaf := self._next_leaf(knit_tree)) is not None:
            if not leaf.is_leaf():
                raise ValueError("Chose a non-leaf node")
            self._bisect(leaf)

    def _bisect(self, node: KnitTree) -> None:
        graph = node._circuit_graph.graph().subgraph(node.node_subset).to_undirected()
        left_subset = girvan_newman_bisection(graph)
        node.divide(left_subset)


class QPUSizeBisectionOptimizer(BisectionOptimizer):
    def __init__(self, qpu_size: int) -> None:
        self._qpu_size = qpu_size
        super().__init__()

    @staticmethod
    def _num_qubits(node_set: set[CircuitGraphNode]) -> int:
        return len({node.qubit for node in node_set})

    def _next_leaf(self, node: KnitTree) -> KnitTree | None:
        for leaf in node.leafs():
            if self._num_qubits(leaf.node_subset) > self._qpu_size:
                return leaf
        return None


def girvan_newman_bisection(graph: nx.Graph) -> set:
    """Bisects a graph using the Girvan-Newman algorithm.

    Args:
        graph (nx.Graph): The graph to cut.

    Returns:
        set: One of the two subsets.
    """

    # get the edge which is most influential to the graph's connectivity
    def central_edge() -> tuple[int, int]:
        centrality = nx.edge_betweenness_centrality(graph, weight="weight")
        return max(centrality, key=centrality.get)

    while nx.number_connected_components(graph) < 2:
        cur_edge = central_edge()
        while nx.has_path(graph, *cur_edge):
            graph.remove_edge(*cur_edge)
            if not nx.has_path(graph, *cur_edge):
                break
            cur_edge = central_edge()

    return next(nx.connected_components(graph))


# if __name__ == "__main__":
#     from qiskit import QuantumCircuit
#     from qvm.compiler.circuit_graph import subgraph_to_circuit_tensor

#     circuit = QuantumCircuit(4)
#     circuit.h(0)
#     circuit.cx(0, 1)
#     circuit.cx(1, 2)
#     circuit.cx(2, 3)

#     circuit_graph = CircuitGraph(circuit)
#     knit_tree = KnitTree(circuit_graph, set(circuit_graph.graph().nodes))

#     optimizer = QPUSizeBisectionOptimizer(2)
#     optimizer.optimize(knit_tree)
#     print(knit_tree.contraction_cost())

#     for leaf in knit_tree.leafs():
#         tens = subgraph_to_circuit_tensor(circuit_graph, leaf.node_subset)
#         print(tens.circuit)
#         print()


import quimb.tensor as qtn
import numpy as np

A = qtn.Tensor(np.random.rand(4), inds=('k0',))
B = qtn.Tensor(np.random.rand(4, 4), inds=('k0','k1'))
C = qtn.Tensor(np.random.rand(4), inds=('k1',))

tn = qtn.TensorNetwork([A, B, C])
print(tn.contraction_cost())
