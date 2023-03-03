from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.transpiler import CouplingMap


def fit_to_coupling_map(
    circuit: QuantumCircuit, coupling_map: CouplingMap
) -> QuantumCircuit:
    if circuit.num_qubits != coupling_map.size():
        raise ValueError(
            "The circuit has a different number of qubits than the coupling map."
        )

    def _qubit_index(qubit: Qubit):
        return circuit.find_bit(qubit).index

    for cinstr in circuit.data:
        if len(cinstr.qubits) == 1:
            continue
        if len(cinstr.qubits) > 2:
            raise ValueError("Only 1- and 2-qubit gates are supported.")
        qubit_index1, qubit_index2 = [circuit.find_bit(qubit).index for qubit in cinstr.qubits]
        if (qubit_index1, qubit_index2) not in coupling_map.get_edges():
            pass
            