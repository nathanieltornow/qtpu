from abc import ABC, abstractmethod

from qvm.circuit import DistributedCircuit


class Compiler(ABC):
    @abstractmethod
    def run(self, vc: DistributedCircuit) -> DistributedCircuit:
        pass
