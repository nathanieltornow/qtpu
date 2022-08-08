from typing import List, Optional, Sequence
import qiskit as qs

from qvm.circuit.operation import Operation


class Circuit:
    _operations: List[Operation]
    _num_qubits: int
    _num_clbits: int

    def __init__(
        self,
        *operations: Operation,
        num_qubits: Optional[int] = None,
        num_clbits: Optional[int] = None
    ) -> None:
        pass

    @staticmethod
    def from_qiskit(circuit: qs.QuantumCircuit) -> "Circuit":
        pass

    def qiskit_circuit(self) -> qs.QuantumCircuit:
        pass

    @property
    def operations(self) -> List[Operation]:
        return self._operations.copy()

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        return self._num_clbits
