from qiskit.circuit import QuantumCircuit


def initial_layout_from_transpiled_circuit(
    original_circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit
) -> list[int]:
    if transpiled_circuit._layout is None:
        raise ValueError("Circuit has no layout.")

    initial_layout = [0] * original_circuit.num_qubits

    layout = transpiled_circuit._layout.initial_layout.get_virtual_bits()

    for i, qubit in enumerate(original_circuit.qubits):
        initial_layout[i] = layout[qubit]

    return initial_layout
