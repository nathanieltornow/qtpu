from qiskit.circuit import QuantumCircuit


def initial_layout_from_transpiled_circuit(circuit: QuantumCircuit) -> list[int]:
    layout = circuit._layout
    if layout is None:
        raise ValueError("Circuit has no layout")
    return layout.get_virtual_bits()
