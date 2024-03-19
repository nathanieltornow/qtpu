from typing import Optional, Callable

import numpy as np
import networkx as nx
from qiskit.circuit import QuantumCircuit

from qvm.instructions import VirtualBinaryGate

from .graph import CircuitGraph
from .girvan_newman import girvan_newman_cut_graph
from ._estimator import SuccessEstimator


class ContractionTree:
    def __init__(self, graph: nx.Graph) -> None:
        self._graph = graph
        self._left: Optional["ContractionTree"] = None
        self._right: Optional["ContractionTree"] = None
        self._cut_edges: set[tuple[int, int]] = set()

    def is_leaf(self) -> bool:
        return self._left is None and self._right is None

    @property
    def left(self) -> Optional["ContractionTree"]:
        return self._left

    @property
    def right(self) -> Optional["ContractionTree"]:
        return self._right

    def removed_edges(self) -> set[tuple[int, int]]:
        if self.is_leaf():
            return self._cut_edges
        return (
            self._cut_edges | self._left.removed_edges() | self._right.removed_edges()
        )

    def contraction_cost(self) -> int:
        if self.is_leaf():
            return 0

        this_contract_cost = int(
            np.prod([self._graph[u][v].get("weight", 1) for u, v in self._cut_edges])
        )

        return this_contract_cost + (
            self._left.contraction_cost()
            * self._right.contraction_cost()
            * self._right.contraction_cost()
        )

    def leafs(self) -> list["ContractionTree"]:
        if self.is_leaf():
            return [self]
        return self._left.leafs() + self._right.leafs()

    def bisect(
        self, bisect_func: Callable[[nx.Graph], tuple[set[int], set[int]]] | None = None
    ) -> None:
        if bisect_func is None:
            bisect_func = _default_bisect

        if not self.is_leaf():
            largest_leaf = max(
                self.leafs(), key=lambda leaf: leaf._graph.number_of_nodes()
            )
            return largest_leaf.bisect(bisect_func=bisect_func)

        left, right = bisect_func(self._graph)

        for u, v in self._graph.edges():
            if u in left and v in right or u in right and v in left:
                self._cut_edges.add((u, v))

        self._left, self._right = (
            ContractionTree(nx.subgraph(self._graph, left)),
            ContractionTree(nx.subgraph(self._graph, right)),
        )


def _default_bisect(graph: nx.Graph) -> tuple[set[int], set[int]]:
    graph = graph.copy()
    girvan_newman_cut_graph(graph, 2)
    con = list(nx.connected_components(graph))
    assert len(con) == 2
    return con[0], con[1]


def contraction_tree_cut_circuit(
    circuit: QuantumCircuit,
    success_estimator: SuccessEstimator,
    max_contraction_cost: int = 1000,
    alpha: float = 0.5,
    max_iters: int = 100,
):
    if any(isinstance(instr.operation, VirtualBinaryGate) for instr in circuit):
        raise ValueError("Circuit already contains virtual gates")

    circuit_graph = CircuitGraph(circuit)
    contraction_tree = ContractionTree(circuit_graph.get_nx_graph())

    current_score = (1 - alpha) * success_estimator.estimate(circuit)

    for _ in range(max_iters):
        contraction_tree.bisect()
        new_circuit = circuit_graph.generate_circuit(contraction_tree.removed_edges())

        success_est = success_estimator.estimate(new_circuit)
        if success_est <= 0.0:
            continue

        knit_cost = contraction_tree.contraction_cost() / max_contraction_cost
        if knit_cost > 1.0:
            raise ValueError("Contraction cost too high")

        score = alpha * (1 - knit_cost) + (1 - alpha) * success_est

        if score > current_score:
            current_score = score
            circuit = new_circuit
        else:
            break

    return circuit
