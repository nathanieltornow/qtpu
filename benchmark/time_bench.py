from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qvm.stack.decomposer._decomposer import Decomposer


def run_benchmarks(
    decomposer: Decomposer, circuits: list[QuantumCircuit], filepath: str
):
    sim = AerSimulator("matrix_product_state")
