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


