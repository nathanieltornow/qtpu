from abc import ABC, abstractmethod
import itertools
from typing import List, Type
from qiskit.circuit.quantumcircuit import (
    QuantumCircuit,
    Instruction,
)

from qvm.result import Result


class VirtualBinaryGate(Instruction, ABC):

    _ids = itertools.count(0)
    is_fragmenting: bool

    def __init__(self, original_gate: Instruction, is_fragmenting: bool = True):
        self.id = next(self._ids)
        if original_gate.num_qubits != 2 or original_gate.num_clbits > 0:
            raise ValueError("The original gate must be a binary gate.")
        if type(original_gate) != self.original_gate_type():
            raise ValueError(
                f"Cannot virtualize {type(original_gate)} with virtual gate for {self.original_gate_type()}"
            )
        self.is_fragmenting = is_fragmenting
        super().__init__(f"virtual_{original_gate.name}", 2, 0, original_gate.params)

    def __eq__(self, other):
        return super().__eq__(other) and self.id == other.id

    def __hash__(self):
        return self.id

    @abstractmethod
    def original_gate_type(self) -> Type[Instruction]:
        """
        Returns the type of the original gate.
        """
        pass

    @abstractmethod
    def configure(self) -> List[QuantumCircuit]:
        pass

    @abstractmethod
    def knit(self, results: List[Result]) -> Result:
        pass

    def _define(self):
        qc = QuantumCircuit(2)
        qc.append(self.original_gate_type()(*self.params), [0, 1])
        self._definition = qc
