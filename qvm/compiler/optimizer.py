import abc
import itertools

import networkx as nx
from networkx.algorithms.community import girvan_newman

from qvm.graph import CircuitGraph, CircuitGraphNode
from qvm.compiler._compression import (
    CompressedNode,
    compress_qubits,
    compress_nothing,
    remove_1q_gates,
    decompress_nodes,
)
from qvm.compiler.contraction_tree import GraphContractionTree


class Optimizer(abc.ABC):
    def __init__(self, compression_method: str | None = None) -> None:
        self._compression_method = compression_method

    def optimize(self, circuit_graph: CircuitGraph) -> list[set[CircuitGraphNode]]:
        match self._compression_method:
            case "qubits":
                compressed_graph = compress_qubits(circuit_graph)
            case "rm_1q":
                compressed_graph = remove_1q_gates(circuit_graph)
            case None:
                compressed_graph = compress_nothing(circuit_graph)
            case _:
                raise ValueError("Invalid compression method")

        contraction_tree = GraphContractionTree(compressed_graph)
        self._optimize(contraction_tree)
        return [
            decompress_nodes(set(leaf.graph.nodes)) for leaf in contraction_tree.leafs()
        ]

    @abc.abstractmethod
    def _optimize(self, tree: GraphContractionTree) -> None: ...


class GreedyOptimizer(Optimizer, abc.ABC):
    def __init__(self, compression_method: str | None = None) -> None:
        super().__init__(compression_method)

    def _bisect(
        self, graph: nx.Graph
    ) -> tuple[set[CompressedNode], set[CompressedNode]]:
        components = next(girvan_newman(graph))
        return components[0], set(itertools.chain.from_iterable(components[1:]))

    @abc.abstractmethod
    def _next_leaf(self, tree: GraphContractionTree) -> GraphContractionTree | None: ...

    def _optimize(self, tree: GraphContractionTree) -> None:
        while (leaf := self._next_leaf(tree)) is not None:
            if not leaf.is_leaf():
                raise ValueError("Chose a non-leaf node")
            A, _ = self._bisect(leaf.graph)
            leaf.divide(A)


class NumQubitsOptimizer(GreedyOptimizer):
    def __init__(self, qpu_size: int, compression_method: str | None = None) -> None:
        self._qpu_size = qpu_size
        super().__init__(compression_method=compression_method)

    def _num_qubits(self, node_set: set[CompressedNode]) -> int:
        return len(set(itertools.chain.from_iterable(node.qubits for node in node_set)))

    def _next_leaf(self, tree: GraphContractionTree) -> GraphContractionTree | None:
        for leaf in tree.leafs():
            if self._num_qubits(leaf.graph.nodes) > self._qpu_size:
                return leaf
        return None
