import abc

from qiskit.circuit import QuantumCircuit, Qubit


class CutStrategy(abc.ABC):
    @abc.abstractmethod
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass


def cut(circuit: QuantumCircuit, strategy: CutStrategy) -> QuantumCircuit:
    return strategy.run(circuit)
