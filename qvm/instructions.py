import abc

import numpy as np
from qiskit.circuit import Gate, Instruction, QuantumCircuit, Barrier, Parameter, Qubit


class VirtualBinaryGate(Barrier, abc.ABC):
    def __init__(self, original_gate: Gate):
        self._original_gate = original_gate
        super().__init__(
            num_qubits=2,
            label=f"v_{original_gate.name}",
        )

    @property
    def original_gate(self) -> Gate:
        return self._original_gate

    @property
    def num_instantiations(self) -> int:
        return len(self.instantiations())

    @abc.abstractmethod
    def instantiations(self) -> list[QuantumCircuit]:
        ...

    @abc.abstractmethod
    def coefficients_1d(self) -> np.ndarray:
        pass

    def instantiations_qubit0(self) -> list[QuantumCircuit]:
        return [inst[0] for inst in self.instantiations()]

    def instantiations_qubit1(self) -> list[QuantumCircuit]:
        return [inst[1] for inst in self.instantiations()]

    def coefficients_2d(self) -> np.ndarray:
        return np.diag(self.coefficients_1d())


class InstantiableInstruction(Instruction):
    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        param_label: str,
        instantiations: list[QuantumCircuit],
    ):
        assert all(circ.num_qubits <= num_qubits for circ in instantiations)
        assert all(circ.num_clbits <= num_clbits for circ in instantiations)
        super().__init__(
            name="inst",
            num_qubits=num_qubits,
            num_clbits=num_clbits,
            params=[Parameter(param_label)],
        )
        self._instantiations = instantiations

    @property
    def param(self) -> Parameter:
        return self.params[0]

    @staticmethod
    def from_virtual_gate(
        virtual_gate: VirtualBinaryGate, vgate_idx: int
    ) -> "InstantiableInstruction":
        return InstantiableInstruction(
            num_qubits=2,
            num_clbits=1,
            param_label=f"vgate_{vgate_idx}",
            instantiations=virtual_gate.instantiations(),
        )

    @staticmethod
    def from_virtual_gate_divided(
        virtual_gate: VirtualBinaryGate, vgate_idx: int
    ) -> tuple["InstantiableInstruction", "InstantiableInstruction"]:
        return (
            InstantiableInstruction(
                num_qubits=1,
                num_clbits=1,
                param_label=f"vgate_{vgate_idx}_0",
                instantiations=virtual_gate.instantiations_qubit0(),
            ),
            InstantiableInstruction(
                num_qubits=1,
                num_clbits=1,
                param_label=f"vgate_{vgate_idx}_1",
                instantiations=virtual_gate.instantiations_qubit1(),
            ),
        )

    @property
    def num_instantiations(self) -> int:
        return len(self._instantiations)

    def _define(self):
        inst_idx = int(self.params[0])
        self._definition = self._instantiations[inst_idx]
