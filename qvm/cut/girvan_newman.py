import networkx as nx
from qiskit.circuit import QuantumCircuit

from .graph import CircuitGraph


def girvan_newman_cut_circuit(
    circuit: QuantumCircuit, num_fragments: int
) -> QuantumCircuit:
    """Cuts a circuit into fragments using the Girvan-Newman algorithm.

    Args:
        circuit (QuantumCircuit): The circuit to cut.
        num_fragments (int): The number of fragments to cut the circuit into.

    Returns:
        QuantumCircuit: The cut circuit.
    """
    circuit_graph = CircuitGraph(circuit)
    graph = circuit_graph.get_nx_graph()

    cut_edges = girvan_newman_cut_graph(graph, num_fragments)

    return circuit_graph.generate_circuit(cut_edges)


def girvan_newman_cut_graph(
    graph: nx.Graph, num_fragments: int
) -> list[tuple[int, int]]:
    """Cuts a graph into fragments using the Girvan-Newman algorithm.

    Args:
        graph (nx.Graph): The graph to cut.
        num_fragments (int): The number of fragments to cut the graph into.

    Returns:
        list[tuple[int, int]]: The cut edges.
    """

    # get the edge which is most influential to the graph's connectivity
    def central_edge() -> tuple[int, int]:
        centrality = nx.edge_betweenness_centrality(graph, weight="weight")
        return max(centrality, key=centrality.get)

    cut_edges = []
    while nx.number_connected_components(graph) < num_fragments:
        cur_edge = central_edge()
        while nx.has_path(graph, *cur_edge):
            graph.remove_edge(*cur_edge)
            cut_edges.append(cur_edge)
            if not nx.has_path(graph, *cur_edge):
                break
            cur_edge = central_edge()

    return cut_edges
