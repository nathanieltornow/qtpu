import networkx as nx


class CircuitGraph:
    def __init__(self) -> None:
        pass

    @property
    def graph(self) -> nx.Graph:
        pass


class SubCircuitGraph:
    pass


class KnitTree:
    def __init__(self, graph: nx.Graph) -> None:
        self._graph = graph
        pass

    @property
    def left(self) -> "KnitTree" | None:
        pass

    @property
    def right(self) -> "KnitTree" | None:
        pass

    def edges_between(self) -> set[tuple[int, int]]:
        pass

    def knit_cost(self) -> int:
        pass

    def divide(self, left_nodes: set[int]) -> None:
        pass
