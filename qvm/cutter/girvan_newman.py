from typing import Iterable
import networkx as nx

from ._cutter import Cutter
from ._graphs import TNGraph


class GirvanNewmanCutter(Cutter):
    def __init__(self, max_cost: int) -> None:
        super().__init__()
        self._max_cost = max_cost

    def _cut(self, tn_graph: TNGraph) -> set[tuple[int, int]]:
        cut_edges = set()
        for mv_edges in girvan_newman(tn_graph):
            if self._cut_cost(tn_graph, cut_edges | set(mv_edges)) > self._max_cost:
                break
            cut_edges.update(mv_edges)
        return cut_edges


def girvan_newman(G: TNGraph) -> Iterable[tuple[int, int]]:
    """Finds cuts in a graph using the Girvan–Newman method."""
    # If the graph is already empty, simply return its connected
    # components.
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return

    def most_valuable_edge(G):
        """Returns the edge with the highest betweenness centrality
        in the graph `G`.

        """
        # We have guaranteed that the graph is non-empty, so this
        # dictionary will never be empty.
        betweenness = nx.edge_betweenness_centrality(G, weight="weight")
        return max(betweenness, key=betweenness.get)

    # The copy of G here must include the edge weight data.
    g = G.copy()
    # Self-loops must be removed because their removal has no effect on
    # the connected components of the graph.
    g.remove_edges_from(nx.selfloop_edges(g))
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)


def _without_most_central_edges(G, most_valuable_edge) -> list[tuple[int, int]]:
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components

    removed_edges = []

    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        removed_edges.append((edge[0], edge[1]))
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return removed_edges
