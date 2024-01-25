import pymetis
import networkx as nx
from qiskit.circuit import QuantumCircuit

from .graph import CircuitGraph


def metis_cut_circuit(circuit: QuantumCircuit, num_fragments: int) -> QuantumCircuit:
    circuit_graph = CircuitGraph(circuit)
    graph = circuit_graph.get_nx_graph()

    cut_edges = metis_cut_graph(graph, num_fragments)

    return circuit_graph.generate_circuit(cut_edges)


def metis_cut_graph(graph: nx.Graph, num_fragments: int) -> list[tuple[int, int]]:
    """Cut a graph into fragments using the METIS algorithm.

    Args:
        graph (nx.Graph): The graph to cut.
        num_fragments (int): The number of fragments to cut the graph into.

    Returns:
        list[tuple[int, int]]: The cut edges.
    """
    xadj, adjncy, adjwgt = _networkx_to_adjacency_structure(graph)

    _, membership = pymetis.part_graph(
        num_fragments, xadj=xadj, adjncy=adjncy, eweights=adjwgt
    )

    cut_edges = []
    for u, v in graph.edges:
        if membership[u] != membership[v]:
            cut_edges.append((u, v))
            graph.remove_edge(u, v)
    return cut_edges


def _networkx_to_adjacency_structure(graph: nx.Graph):
    # TODO there is some bug in here
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
