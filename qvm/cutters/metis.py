import networkx as nx
import pymetis

from ._cutter import Cutter


class MetisCutter(Cutter):
    def __init__(self, num_fragments: int) -> None:
        super().__init__()
        self._num_fragments = num_fragments

    def _cut(self, graph: nx.Graph) -> list[tuple[int, int]]:
        return cut_graph_using_metis(graph, self._num_fragments)


def cut_graph_using_metis(
    cut_graph: nx.Graph, num_fragments: int
) -> list[tuple[int, int]]:
    xadj, adjncy, adjwgt = _networkx_to_adjacency_structure(cut_graph)

    _, membership = pymetis.part_graph(
        num_fragments, xadj=xadj, adjncy=adjncy, eweights=adjwgt
    )

    cut_edges = []
    for u, v in cut_graph.edges:
        if membership[u] != membership[v]:
            cut_edges.append((u, v))
    return cut_edges


def _networkx_to_adjacency_structure(graph: nx.Graph):
    xadj = []
    adjncy = []
    adjwgt = []

    # Nodes sorted in ascending order
    nodes = sorted(graph.nodes())

    # Iterate over nodes to create xadj, adjncy, and adjwgt
    for node in nodes:
        neighbors = list(graph.neighbors(node))

        # Append the start index of adjacency list for the current node
        xadj.append(len(adjncy))

        # Append the neighbors of the current node to adjncy
        adjncy.extend(neighbors)

        # Append the weights of the edges to adjwgt
        weights = [graph[node][neighbor].get("weight", 1) for neighbor in neighbors]
        adjwgt.extend(weights)

    # Append the end index for the last node
    xadj.append(len(adjncy))

    return xadj, adjncy, adjwgt
