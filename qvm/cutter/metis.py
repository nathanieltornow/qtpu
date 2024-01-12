# import networkx as nx

# from ._cutter import PortGraphCutter, QubitGraphCutter, TNCutter


# class MetisTNCutter(TNCutter):
#     def __init__(self, num_fragments: int) -> None:
#         self._num_fragments = num_fragments
#         super().__init__()

#     def _cut_tn(self, tn_graph: nx.Graph) -> list[tuple[int, int]]:
#         return cut_graph_using_metis(tn_graph, self._num_fragments)


# class MetisQubitGraphCutter(QubitGraphCutter):
#     def __init__(self, num_fragments: int) -> None:
#         self._num_fragments = num_fragments
#         super().__init__()

#     def _cut_qubit_graph(self, cut_graph: nx.Graph) -> list[tuple[int, int]]:
#         return cut_graph_using_metis(cut_graph, self._num_fragments)


# class MetisPortGraphCutter(PortGraphCutter):
#     def __init__(self, num_fragments: int) -> None:
#         self._num_fragments = num_fragments
#         super().__init__()

#     def _cut_portgraph(self, cut_graph: nx.Graph) -> list[tuple[int, int]]:
#         return cut_graph_using_metis(cut_graph, self._num_fragments)


# def cut_graph_using_metis(
#     cut_graph: nx.Graph, num_fragments: int
# ) -> list[tuple[int, int]]:
#     try:
#         import pymetis
#     except ImportError as e:
#         raise ImportError("MetisCutter requires pymetis") from e

#     xadj, adjncy, adjwgt = _networkx_to_adjacency_structure(cut_graph)

#     _, membership = pymetis.part_graph(
#         num_fragments, xadj=xadj, adjncy=adjncy, eweights=adjwgt
#     )

#     cut_edges = []
#     for u, v in cut_graph.edges:
#         if membership[u] != membership[v]:
#             cut_edges.append((u, v))
#     return cut_edges


# def _networkx_to_adjacency_structure(graph: nx.Graph):
#     xadj = []
#     adjncy = []
#     adjwgt = []

#     # Nodes sorted in ascending order
#     nodes = sorted(graph.nodes())

#     # Iterate over nodes to create xadj, adjncy, and adjwgt
#     for node in nodes:
#         neighbors = list(graph.neighbors(node))

#         # Append the start index of adjacency list for the current node
#         xadj.append(len(adjncy))

#         # Append the neighbors of the current node to adjncy
#         adjncy.extend(neighbors)

#         # Append the weights of the edges to adjwgt
#         weights = [graph[node][neighbor].get("weight", 1) for neighbor in neighbors]
#         adjwgt.extend(weights)

#     # Append the end index for the last node
#     xadj.append(len(adjncy))

#     return xadj, adjncy, adjwgt
