from abc import ABC, abstractmethod
import itertools
from typing import List, Optional, Type
from qiskit.circuit.quantumcircuit import QuantumCircuit, Instruction

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


# class PartialVirtualGate(VirtualGate):
#     _original_virtual_gate: VirtualGate
#     _qubit_indices: List[int]

#     def __init__(self, virtual_gate: VirtualGate, qubit_indices: List[int]):
#         self._original_virtual_gate = virtual_gate
#         self._qubit_indices = qubit_indices
#         super().__init__(
#             virtual_gate.name,
#             len(qubit_indices),
#             virtual_gate.num_clbits,
#             virtual_gate.params,
#         )

#     @staticmethod
#     def _circuit_on_qubit_indices(
#         circuit: QuantumCircuit, qubit_indices: List[int]
#     ) -> QuantumCircuit:
#         new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
#         [
#             new_circuit.append(instr)
#             for instr in circuit.data
#             if instr.qubits[0] in qubit_indices
#         ]
#         return new_circuit

#     def configure(self) -> List[QuantumCircuit]:
#         return [
#             self._circuit_on_qubit_indices(config, self._qubit_indices)
#             for config in self._original_virtual_gate.configure()
#         ]

#     def knit(self, results: List[Result]) -> Result:
#         raise NotImplementedError("cannot knit partial virtual gate")
