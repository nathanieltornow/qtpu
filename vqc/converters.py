import itertools

import networkx as nx
from qiskit.circuit import Barrier, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit


def dag_to_connectivity_graph(dag: DAGCircuit) -> nx.Graph:
    graph = nx.Graph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    graph.add_nodes_from(dag.qubits)
    for node in dag.op_nodes():
        if isinstance(node.op, Barrier):
            continue
        if len(node.qargs) >= 2:
            for qarg1, qarg2 in itertools.combinations(node.qargs, 2):
                if not graph.has_edge(qarg1, qarg2):
                    graph.add_edge(qarg1, qarg2, weight=0)
                graph[qarg1][qarg2]["weight"] += 1
    return graph


def circuit_to_connectivity_graph(circuit: QuantumCircuit) -> nx.Graph:
    dag = circuit_to_dag(circuit)
    return dag_to_connectivity_graph(dag)
