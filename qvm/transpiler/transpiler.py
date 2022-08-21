from abc import ABC, abstractmethod
from typing import List, Set

from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator

from qvm.circuit import VirtualCircuitBase, FragmentedVirtualCircuit
from .transpiled_circuit import TranspiledFragmentedCircuit


class DecompositionTranspiler(ABC):
    @abstractmethod
    def run(self, circuit: FragmentedVirtualCircuit) -> None:
        pass


class FragmentToDeviceTranspiler(ABC):
    _backends: List[Backend]

    def __init__(self, *available_backends: Backend) -> None:
        self._backends = list(available_backends)
        if len(self._backends) == 0:
            raise ValueError("No available backends given")

    @abstractmethod
    def run(self, circuit: TranspiledFragmentedCircuit) -> TranspiledFragmentedCircuit:
        pass


class LayoutTranspiler(ABC):
    @abstractmethod
    def run(self, virtual_circuit: VirtualCircuitBase, backend: Backend) -> None:
        pass
