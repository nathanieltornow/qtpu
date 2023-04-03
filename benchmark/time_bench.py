from qiskit.circuit import QuantumCircuit
from qvm.stack.decomposer._decomposer import Decomposer

from qiskit_aer import AerSimulator


def run_benchmarks(
    decomposer: Decomposer, circuits: list[QuantumCircuit], filepath: str
):
    sim = AerSimulator("matrix_product_state")
