from typing import Iterator, Optional

from ..circuit_graph import CircuitGraph, CircuitGraphNode


class KnitTree:
    def __init__(
        self, circuit_graph: CircuitGraph, node_subset: set[CircuitGraphNode]
    ) -> None:
        self._circuit_graph = circuit_graph
        self._node_subset = node_subset
        self._left_child: KnitTree | None = None
        self._right_child: KnitTree | None = None
        self._own_contraction_cost = None

    @property
    def circuit_graph(self) -> CircuitGraph:
        return self._circuit_graph

    @property
    def node_subset(self) -> set[CircuitGraphNode]:
        return self._node_subset

    @property
    def left_child(self) -> Optional["KnitTree"]:
        return self._left_child

    @property
    def right_child(self) -> Optional["KnitTree"]:
        return self._right_child

    def traverse(self) -> Iterator["KnitTree"]:
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            if node.right_child is not None:
                stack.append(node.right_child)
            if node.left_child is not None:
                stack.append(node.left_child)

    def leafs(self) -> Iterator["KnitTree"]:
        for node in self.traverse():
            if node.is_leaf():
                yield node

    def contraction_cost(self) -> int:
        if self.is_leaf():
            return 0

        if self._own_contraction_cost is None:
            graph = self._circuit_graph.graph()
            left_subset = self._left_child.node_subset
            right_subset = self._right_child.node_subset

            cost = 1
            for u, v, weight in graph.edges(data="weight"):
                if (u in left_subset and v in right_subset) or (
                    u in right_subset and v in left_subset
                ):
                    cost *= weight

            self._own_contraction_cost = cost

        return (
            self._own_contraction_cost
            + self._left_child.contraction_cost()
            + self._right_child.contraction_cost()
        )

    def is_leaf(self) -> bool:
        return self._left_child is None and self._right_child is None

    def divide(self, left_subset: set[CircuitGraphNode]) -> None:
        if not self.is_leaf():
            raise ValueError("Can only divide leaves")

        self._left_child = KnitTree(self._circuit_graph, left_subset)
        self._right_child = KnitTree(
            self._circuit_graph, self._node_subset - left_subset
        )
