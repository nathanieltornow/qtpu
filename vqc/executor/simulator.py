from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from vqc.prob_distr import Counts
from .executor import Executor


class Simulator(Executor):
    def execute(
        self, sampled_circuits: dict[str, list[QuantumCircuit]]
    ) -> dict[str, list[Counts]]:
        results = {}
        for name, frag_circs in sampled_circuits.items():
            results[name] = (
                AerSimulator()
                .run([circ.decompose() for circ in frag_circs])
                .result()
                .get_counts()
            )
        return results
