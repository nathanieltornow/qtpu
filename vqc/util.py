import itertools

import networkx as nx
from qiskit.circuit import Barrier, QuantumCircuit, QuantumRegister
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


def circuit_on_index(circuit: QuantumCircuit, index: int) -> QuantumCircuit:
    qreg = QuantumRegister(1)
    qubit = circuit.qubits[index]
    circ = QuantumCircuit(qreg, *circuit.cregs)
    for instr in circuit.data:
        if len(instr.qubits) == 1 and instr.qubits[0] == qubit:
            circ.append(instr.operation, (qreg[0],), instr.clbits)
    return circ
