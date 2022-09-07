from abc import ABC, abstractmethod

from qvm.circuit import VirtualCircuit


class Compiler(ABC):
    @abstractmethod
    def run(self, vc: VirtualCircuit) -> VirtualCircuit:
        pass
