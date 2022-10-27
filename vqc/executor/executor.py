from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit

from vqc.prob_distr import Counts


class Executor(ABC):
    @abstractmethod
    def execute(
        self, sampled_circuits: dict[str, list[QuantumCircuit]]
    ) -> dict[str, list[Counts]]:
        pass
