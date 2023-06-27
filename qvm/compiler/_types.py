import abc

from qiskit.circuit import QuantumCircuit

from qvm.virtualizer import Virtualizer


class VirtualizationCompiler(abc.ABC):
    @abc.abstractmethod
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass
