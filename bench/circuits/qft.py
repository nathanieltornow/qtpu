from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFT


def qft(num_qubits: int, approx: int = 0) -> QuantumCircuit:
    circuit = QFT(num_qubits, approximation_degree=approx, do_swaps=False)
    circuit.measure_all()
    return circuit.decompose()