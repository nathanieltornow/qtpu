from abc import ABC, abstractmethod
import itertools
from typing import List, Optional, Type
from qiskit.circuit.quantumcircuit import (
    QuantumCircuit,
    Instruction,
    CircuitInstruction,
)

from qvm.result import Result


class VirtualBinaryGate(Instruction, ABC):

    _ids = itertools.count(0)

    def __init__(self, original_gate: Instruction):
        self.id = next(self._ids)
        if original_gate.num_qubits != 2 or original_gate.num_clbits > 0:
            raise ValueError("The original gate must be a binary gate.")
        if type(original_gate) != self.original_gate_type():
            raise ValueError(
                f"Cannot virtualize {type(original_gate)} with virtual gate for {self.original_gate_type()}"
            )
        super().__init__(original_gate.name, 2, 1, original_gate.params)

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


class PartialVirtualGate(Instruction):
    _vgate: VirtualBinaryGate
    _index: int

    def __init__(self, virtual_gate: VirtualBinaryGate, qubit_index: int) -> None:
        self._vgate = virtual_gate
        self._index = qubit_index
        super().__init__(virtual_gate.name, 1, 1, [])

    @staticmethod
    def _circuit_on_qubit_index(
        circuit: QuantumCircuit, qubit_index: int
    ) -> QuantumCircuit:
        new_circuit = QuantumCircuit(1, 1)
        [
            new_circuit.append(
                CircuitInstruction(
                    instr.operation,
                    [new_circuit.qubits[0]],
                    [new_circuit.clbits[0] for _ in instr.clbits],
                )
            )
            for instr in circuit.data
            if circuit.find_bit(instr.qubits[0]).index == qubit_index
        ]
        return new_circuit

    def configure(self) -> List[QuantumCircuit]:
        return [
            self._circuit_on_qubit_index(circuit, self._index)
            for circuit in self._vgate.configure()
        ]
