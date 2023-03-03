import networkx as nx

from qiskit.circuit import QuantumCircuit, Qubit


def circuit_to_simple_dag(circuit: QuantumCircuit) -> nx.DiGraph:
    graph = nx.DiGraph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    graph.add_nodes_from(range(len(circuit.data)))

    def _next_operation_on_qubit(from_index: int, qubit: Qubit) -> int:
        for i, cinstr in circuit.data[from_index + 1 :]:
            if qubit in cinstr.qubits:
                return i
        return -1

    for i, cinstr in circuit.data:
        for qubit in cinstr.qubits:
            next_op = _next_operation_on_qubit(i, qubit)
            if next_op != -1:
                if not graph.has_edge(i, next_op):
                    graph.add_edge(i, next_op, weight=0)
                graph[i][next_op]["weight"] += 1

    return graph
