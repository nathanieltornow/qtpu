from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from vqc.prob_distr import Counts
from .executor import Executor


class Simulator(Executor):
    def __init__(self, shots: int = 10000) -> None:
        self._shots = shots

    def execute(
        self, sampled_circuits: dict[str, list[QuantumCircuit]]
    ) -> dict[str, list[Counts]]:
        results = {}
        for name, frag_circs in sampled_circuits.items():
            results[name] = (
                AerSimulator()
                .run([circ.decompose() for circ in frag_circs], shots=self._shots)
                .result()
                .get_counts()
            )
        return results
