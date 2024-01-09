import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from .cutter import TNCutter


class BisectionCutter(TNCutter):
    def _cut_tn(self, cut_graph: nx.Graph) -> list[tuple[int, int]]:
        A, B = kernighan_lin_bisection(cut_graph)
        cut_edges = []
        for node1, node2 in cut_graph.edges:
            if (node1 in A and node2 in B) or (node1 in B and node2 in A):
                cut_edges.append((node1, node2))
        return cut_edges
