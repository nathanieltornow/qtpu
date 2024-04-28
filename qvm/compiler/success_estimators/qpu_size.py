from qiskit.circuit import QuantumCircuit, Barrier, Measure

from qvm.virtual_circuit import VirtualCircuit
from qvm.instructions import InstantiableInstruction
from qvm.compiler._estimator import SuccessEstimator


class QPUSizeEstimator(SuccessEstimator):
    def __init__(self, max_qpu_size: int) -> None:
        self._max_qpu_size = max_qpu_size
        super().__init__()

    def estimate(self, circuit: QuantumCircuit) -> float:
        vc = VirtualCircuit(circuit)
        max_frag_size = max(len(frag) for frag in vc.fragments)
        return 1.0 if max_frag_size <= self._max_qpu_size else 0.0


class SimpleQPUEstimator(SuccessEstimator):
    def __init__(
        self,
        qpu_size: int,
        error_1q: float = 1e-4,
        error_2q: float = 1e-3,
        error_measure: float = 1e-3,
    ) -> None:
        self._qpu_size = qpu_size
        self._error_1q = error_1q
        self._error_2q = error_2q
        self._error_measure = error_measure
        super().__init__()

    def estimate(self, circuit: QuantumCircuit) -> float:
        vc = VirtualCircuit(circuit)
        max_frag_size = max(len(frag) for frag in vc.fragments)
        if max_frag_size > self._qpu_size:
            return 0.0

        return min(self._eps(frag_circ for frag_circ in vc.fragment_circuits.values()))

    def _eps(self, circuit: QuantumCircuit) -> float:
        fid = 1.0
        for instr in circuit:
            op = instr.operation

            if isinstance(op, Barrier):
                continue

            if isinstance(op, Measure):
                fid *= 1 - self._error_measure

            elif isinstance(op, InstantiableInstruction):
                fid *= 1 - self._error_measure

            elif op.num_qubits == 1:
                fid *= 1 - self._error_1q

            elif op.num_qubits == 2:
                fid *= 1 - self._error_2q

            else:
                raise ValueError(f"Unsupported operation: {op}")

        return fid
