from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_aer import AerSimulator

from qvm.runtime._types import Counts
from qvm.runtime.runner import Runner


class SimRunner(Runner):
    def __init__(self) -> None:
        self._backend = AerSimulator()
        super().__init__()

    def run(
        self,
        circuit: QuantumCircuit,
        arg_batch: list[dict[Parameter, float]],
        shots: int = 10000,
    ) -> list[Counts]:
        circuits = [circuit.assign_parameters(args).decompose() for args in arg_batch]
        counts = self._backend.run(circuits, shots=shots).result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return counts
