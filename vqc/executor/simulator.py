from typing import Dict, List

from qiskit.circuit import QuantumCircuit
from qiskit.providers.models import BackendConfiguration
from qiskit_aer import AerSimulator

from vqc.prob_distr import Counts
from vqc.types import Executor


class Simulator(Executor):
    def __init__(
        self, backend_configuration: BackendConfiguration = None, shots: int = 10000
    ) -> None:
        self._shots = shots
        self._simulator_instance = (
            AerSimulator()
            if backend_configuration is None
            else AerSimulator.from_backend(backend_configuration)
        )

    def execute(
        self, sampled_circuits: Dict[str, List[QuantumCircuit]]
    ) -> Dict[str, List[Counts]]:
        results = {}
        for name, frag_circs in sampled_circuits.items():
            results[name] = (
                self._simulator_instance.run(
                    [circ.decompose() for circ in frag_circs], shots=self._shots
                )
                .result()
                .get_counts()
            )
        return results
