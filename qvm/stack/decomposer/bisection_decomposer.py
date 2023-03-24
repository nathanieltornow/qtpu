from qiskit.circuit import QuantumCircuit

from qvm.cut_library.decomposition import bisect_recursive
from qvm.stack._types import QVMLayer

from ._decomposer import Decomposer


class BisectionDecomposer(Decomposer):
    def __init__(self, sub_layer: QVMLayer, num_fragments: int = 2) -> None:
        super().__init__(sub_layer)
        self._num_frags = num_fragments

    def _decompose(self, qernel: QuantumCircuit) -> QuantumCircuit:
        return bisect_recursive(qernel, self._num_frags)
