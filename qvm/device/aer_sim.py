from typing import List

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerProvider

from qvm.prob import ProbDistribution
from .device import Device


class AerSimDevice(Device):
    def __init__(self, backend_name: str = "statevector_simulator"):
        self.backend = AerProvider().get_backend(backend_name)

    def run(self, circuits: List[QuantumCircuit], shots: int) -> List[ProbDistribution]:
        if len(circuits) == 0:
            return []
        circuits = transpile(circuits, self.backend)
        if len(circuits) == 1:
            return [
                ProbDistribution.from_counts(
                    self.backend.run(circuits[0], shots=shots).result().get_counts()
                )
            ]
        return [
            ProbDistribution.from_counts(cnt)
            for cnt in self.backend.run(circuits, shots=shots).result().get_counts()
        ]
