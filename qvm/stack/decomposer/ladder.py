from qiskit.circuit import QuantumCircuit

from qvm.cut_library.decomposition import decompose_ladder
from qvm.stack._types import QVMLayer

from ._decomposer import Decomposer


class LadderDecomposer(Decomposer):
    def __init__(self, sub_layer: QVMLayer, max_fragment_size: int) -> None:
        super().__init__(sub_layer)
        self._max_frag_size = max_fragment_size

    def _decompose(self, qernel: QuantumCircuit) -> QuantumCircuit:
        return decompose_ladder(qernel, self._max_frag_size)
