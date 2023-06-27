from qiskit.circuit import QuantumCircuit
from qvm.compiler._types import VirtualizationCompiler


class BisectionCompiler(VirtualizationCompiler):
    def __init__(self, size_to_reach: int) -> None:
        super().__init__()

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass
