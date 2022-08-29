import itertools
from typing import Dict, List, Set, Tuple
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.converters import circuit_to_dag
import networkx as nx

from qvm.virtual_gate import VirtualBinaryGate


def circuit_to_connectivity_graph(circuit: QuantumCircuit) -> nx.Graph:
    graph = nx.Graph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    graph.add_nodes_from(circuit.qubits)
    for instr in circuit.data:
        if isinstance(instr.operation, VirtualBinaryGate):
            continue

        if len(instr.qubits) >= 2:
            for qubit1, qubit2 in itertools.combinations(instr.qubits, 2):
                if not graph.has_edge(qubit1, qubit2):
                    graph.add_edge(qubit1, qubit2, weight=0)
                graph[qubit1][qubit2]["weight"] += 1
    return graph


def decompose_virtual_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    return circuit.decompose([VirtualBinaryGate])


def deflated_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    dag = circuit_to_dag(circuit)
    qubits = set(qubit for qubit in circuit.qubits if qubit not in dag.idle_wires())

    qreg = QuantumRegister(bits=qubits)
    new_circuit = QuantumCircuit(qreg, *circuit.cregs)
    sorted_qubits = sorted(qubits, key=lambda q: circuit.find_bit(q).index)
    qubit_map: Dict[Qubit, Qubit] = {
        q: new_circuit.qubits[i] for i, q in enumerate(sorted_qubits)
    }
    for circ_instr in circuit.data:
        if set(circ_instr.qubits) <= qubits:
            new_circuit.append(
                circ_instr.operation,
                [qubit_map[q] for q in circ_instr.qubits],
                circ_instr.clbits,
            )
    return new_circuit
