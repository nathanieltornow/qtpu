from qiskit.circuit import QuantumCircuit

from qvm.virtual_circuit import VirtualCircuit
from qvm.cut._estimator import SuccessEstimator


class QPUSizeEstimator(SuccessEstimator):
    def __init__(self, max_qpu_size: int) -> None:
        self._max_qpu_size = max_qpu_size
        super().__init__()

    def estimate(self, circuit: QuantumCircuit) -> float:
        vc = VirtualCircuit(circuit)
        max_frag_size = max(len(frag) for frag in vc.fragments)
        return 1.0 if max_frag_size <= self._max_qpu_size else 0.0
