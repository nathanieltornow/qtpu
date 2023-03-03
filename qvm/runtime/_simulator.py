from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qvm.runtime.sampler import Sampler
from qvm.quasi_distr import QuasiDistr


class SimulatorSampler(Sampler):
    def _sample(self, circuits: list[QuantumCircuit], shots: int) -> list[QuasiDistr]:
        results = AerSimulator().run(circuits, shots=shots).result()
        all_counts = results.get_counts()
        all_counts = [all_counts] if isinstance(all_counts, dict) else all_counts
        return [
            QuasiDistr.from_counts(counts=counts, shots=shots) for counts in all_counts
        ]


def run_on_simulator(circuit: QuantumCircuit, shots: int) -> QuasiDistr:
    return SimulatorSampler().sample([circuit], shots)[0]