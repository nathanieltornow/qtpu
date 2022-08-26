from abc import ABC, abstractmethod
from typing import List

from qiskit.providers import Backend

from qvm.circuit import VirtualCircuitInterface, VirtualCircuit


class QVMTranspiler(ABC):
    _backends: List[Backend]

    def __init__(self, *available_backends: Backend) -> None:
        self._backends = list(available_backends)
        if len(self._backends) == 0:
            raise ValueError("No available backends given")

    @abstractmethod
    def run(self, circuit: VirtualCircuit) -> VirtualCircuit:
        pass


class LayoutTranspiler(ABC):
    @abstractmethod
    def run(self, virtual_circuit: VirtualCircuitInterface, backend: Backend) -> None:
        pass
