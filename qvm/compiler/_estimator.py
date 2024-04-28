import abc

from qiskit.circuit import QuantumCircuit


class SuccessEstimator(abc.ABC):
    @abc.abstractmethod
    def estimate(self, circuit: QuantumCircuit) -> float:
        ...
