from re import S
from typing import List, Optional, Sequence
import qiskit as qs

from qvm.circuit.operation import BinaryGate, Measurement, Operation, UnaryGate


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
        if not num_qubits:
            num_qubits = 0
        if not num_clbits:
            num_clbits = 0
        self._operations = list(operations)
        for op in self.operations:
            if isinstance(op, UnaryGate):
                num_qubits = max(num_qubits, op.qubit + 1)
            elif isinstance(op, BinaryGate):
                num_qubits = max(num_qubits, op.qubit1 + 1, op.qubit2 + 1)
            elif isinstance(op, Measurement):
                num_qubits = max(num_qubits, op.qubit + 1)
                num_clbits = max(num_clbits, op.clbit + 1)
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits

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
