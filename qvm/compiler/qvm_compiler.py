from abc import ABC, abstractmethod
from qvm.circuit import VirtualCircuit


class QVMCompiler(ABC):
    @abstractmethod
    def run(self, virtual_circuit: VirtualCircuit) -> VirtualCircuit:
        pass
