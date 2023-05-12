from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap

from qvm.core.util import circuit_to_qcg


def vroute_furthest_qubits(
    circuit: QuantumCircuit,
    couping_map: CouplingMap,
    initial_layout: list[int],
    max_gate_cuts: int = 4,
) -> QuantumCircuit:
    if not circuit.num_qubits == len(initial_layout):
        raise ValueError("initial_layout must contain all qubits of the circuit.")
    initial_mapping = {
        qubit: initial_layout[i] for i, qubit in enumerate(circuit.qubits)
    }

    qcg = circuit_to_qcg(circuit)

    qubit_distances: list[tuple[Qubit, Qubit, int]]
    for qubit1, qubit2 in qcg.edges:
        qubit_distances.append(
            (
                qubit1,
                qubit2,
                couping_map.distance(initial_mapping[qubit1], initial_mapping[qubit2]),
            )
        )
    qubit_distances.sort(key=lambda x: x[2], reverse=True)
    
    