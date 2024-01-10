import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from ._cutter import PortGraphCutter, QubitGraphCutter, TNCutter


class BisectionTNCutter(TNCutter):
    def _cut_tn(self, cut_graph: nx.Graph) -> list[tuple[int, int]]:
        A, B = kernighan_lin_bisection(cut_graph)
        cut_edges = []
        for node1, node2 in cut_graph.edges:
            if (node1 in A and node2 in B) or (node1 in B and node2 in A):
                cut_edges.append((node1, node2))
        return cut_edges


class BisectionQubitGraphCutter(QubitGraphCutter):
    def _cut_qubit_graph(self, qubit_graph: nx.Graph) -> list[tuple[int, int]]:
        A, B = kernighan_lin_bisection(qubit_graph)
        cut_edges = []
        for node1, node2 in qubit_graph.edges:
            if (node1 in A and node2 in B) or (node1 in B and node2 in A):
                cut_edges.append((node1, node2))
        return cut_edges


class BisectionPortGraphCutter(PortGraphCutter):
    def _cut_portgraph(self, port_graph: nx.DiGraph) -> list[tuple[int, int]]:
        A, B = kernighan_lin_bisection(port_graph)
        cut_edges = []
        for node1, node2 in port_graph.edges:
            if (node1 in A and node2 in B) or (node1 in B and node2 in A):
                cut_edges.append((node1, node2))
        return cut_edges
