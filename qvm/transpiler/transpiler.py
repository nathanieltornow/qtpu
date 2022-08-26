from abc import ABC, abstractmethod
from typing import List

from qiskit.providers import Backend

from qvm.circuit import VirtualCircuitInterface, VirtualCircuit
from qvm.transpiler.transpiled_circuit import TranspiledVirtualCircuit


class QVMTranspiler(ABC):
    @abstractmethod
    def run(self, circuit: VirtualCircuit) -> VirtualCircuit:
        pass


class FragmentToDeviceTranspiler(ABC):
    backends: List[Backend]

    def __init__(self, available_backends: List[Backend]) -> None:
        self.backends = available_backends

    @abstractmethod
    def run(self, circuit: TranspiledVirtualCircuit) -> None:
        pass


class LayoutTranspiler(ABC):
    @abstractmethod
    def run(self, virtual_circuit: VirtualCircuitInterface, backend: Backend) -> None:
        pass
