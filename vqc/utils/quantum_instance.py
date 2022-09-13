from typing import Dict, List, Union

from qiskit.circuit import QuantumCircuit
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance

from vqc.executor import execute
from vqc.prob import ProbDistribution


class VQCQuantumInstance(QuantumInstance):
    def __init__(
        self,
        shots: int = 10000,
    ) -> None:
        super().__init__(
            AerSimulator(),
            shots,
        )
        self.shots = shots

    @staticmethod
    def _counts_to_result(
        counts: List[Dict[str, int]],
        shots: int,
    ) -> Result:
        results = []
        for cnt in counts:
            results.append(
                ExperimentResult(
                    data=ExperimentResultData(
                        counts=cnt,
                    ),
                ),
            )
        return Result("vqc", "0.0.1", 1, 1, True, results=results)

    def transpile(self, circuits, pass_manager=None):
        return circuits

    def execute(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        had_transpiled: bool = False,
    ) -> Result:
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        counts = []
        for circuit in circuits:
            cnt = execute(circuit, self.shots)
            counts.append(cnt)

        return self._counts_to_result(counts, self.shots)
