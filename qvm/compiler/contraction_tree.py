from typing import Optional

import numpy as np
import networkx as nx


class GraphContractionTree:
    def __init__(self, graph: nx.Graph | nx.DiGraph) -> None:
        self._graph = graph
        self._left: Optional["GraphContractionTree"] = None
        self._right: Optional["GraphContractionTree"] = None
        self._between_edges: set[tuple[int, int, int]] = set()

    def is_leaf(self) -> bool:
        return self._left is None and self._right is None

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def left(self) -> Optional["GraphContractionTree"]:
        return self._left

    @property
    def right(self) -> Optional["GraphContractionTree"]:
        return self._right

    @property
    def between_edges(self) -> set[tuple[int, int, int]]:
        return self._between_edges.copy()

    def leafs(self) -> list["GraphContractionTree"]:
        if self.is_leaf():
            return [self]
        return self._left.leafs() + self._right.leafs()

    def divide(self, left_nodes: set) -> None:
        if not self.is_leaf():
            raise ValueError("Cannot divide a non-leaf node (at the moment)")

        assert left_nodes.issubset(self._graph.nodes)

        left = left_nodes
        right = set(self._graph.nodes) - left

        for u, v, weight in self._graph.edges(data="weight"):
            if (u in left and v in right) or (u in right and v in left):
                self._between_edges.add((u, v, weight))

        self._left = GraphContractionTree(self._graph.subgraph(left))
        self._right = GraphContractionTree(self._graph.subgraph(right))

    def contraction_cost(self) -> int:
        if self.is_leaf():
            return 0
        this_contract_cost = int(
            np.prod([weight for _, _, weight in self._between_edges])
        )
        right_contraction_cost = self._right.contraction_cost() * this_contract_cost
        left_contraction_cost = self._left.contraction_cost() * this_contract_cost
        return this_contract_cost + right_contraction_cost + left_contraction_cost
