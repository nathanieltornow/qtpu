from qiskit.circuit import QuantumCircuit

from qvm.cut_library.decomposition import decompose
from qvm.stack._types import QVMLayer

from ._decomposer import Decomposer


class QPUAwareDecomposer(Decomposer):
    def __init__(self, sub_layer: QVMLayer, max_fragment_size: int) -> None:
        super().__init__(sub_layer)
        assert max_fragment_size > 0
        self._max_frag_size = max_fragment_size

    def _decompose(self, qernel: QuantumCircuit) -> QuantumCircuit:
        max_qubits = min(
            max([qpu.num_qubits() for qpu in self._sub_layer.qpus().values()]),
            self._max_frag_size,
        )
        return decompose(qernel, max_qubits)
