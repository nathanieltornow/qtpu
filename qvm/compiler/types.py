import abc

from qiskit.circuit import QuantumCircuit

from qvm.virtual_circuit import VirtualCircuit


class CutCompiler(abc.ABC):
    """A compiler that inserts virtual operations into a circuit."""

    @abc.abstractmethod
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass


class VirtualCircuitCompiler(abc.ABC):
    """
    A compiler that modifies a virtual circuit (e.g. mapping or qubit reuse).
    """

    @abc.abstractmethod
    def run(self, virt: VirtualCircuit) -> None:
        pass
