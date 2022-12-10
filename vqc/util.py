import itertools

import networkx as nx
from qiskit.circuit import QuantumCircuit, QuantumRegister, Barrier


def _circuit_to_connectivity_graph(circuit: QuantumCircuit) -> nx.Graph:
    graph = nx.Graph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    graph.add_nodes_from(circuit.qubits)
    for instr in circuit:
        if isinstance(instr, Barrier):
            continue
        if len(instr.qubits) >= 2:
            for qubit1, qubit2 in itertools.combinations(instr.qargs, 2):
                if not graph.has_edge(qubit1, qubit2):
                    graph.add_edge(qubit1, qubit2, weight=0)
                graph[qubit1][qubit2]["weight"] += 1


def _circuit_on_index(circuit: QuantumCircuit, index: int) -> QuantumCircuit:
    qreg = QuantumRegister(1)
    qubit = circuit.qubits[index]
    circ = QuantumCircuit(qreg, *circuit.cregs)
    for instr in circuit.data:
        if len(instr.qubits) == 1 and instr.qubits[0] == qubit:
            circ.append(instr.operation, (qreg[0],), instr.clbits)
    return circ
