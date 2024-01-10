import abc

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import (  # Barrier,
    Gate,
    Instruction,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
)


class VirtualBinaryGate(Gate, abc.ABC):
    def __init__(
        self,
        name: str,
        params: list,
        label: str | None = None,
    ) -> None:
        super().__init__(name=name, num_qubits=2, params=params, label=label)

    @property
    def num_instantiations(self) -> int:
        return len(self.instantiations())

    @abc.abstractmethod
    def instantiations(self) -> list[QuantumCircuit]:
        ...

    @abc.abstractmethod
    def coefficients_1d(self) -> NDArray[np.float32]:
        pass

    def instantiations_qubit0(self) -> list[QuantumCircuit]:
        return [self._circ_on_qubit(inst, 0) for inst in self.instantiations()]

    def instantiations_qubit1(self) -> list[QuantumCircuit]:
        return [self._circ_on_qubit(inst, 1) for inst in self.instantiations()]

    def coefficients_2d(self) -> NDArray[np.float32]:
        return np.diag(self.coefficients_1d())

    @staticmethod
    def _circ_on_qubit(circ: QuantumCircuit, qubit_index: int) -> QuantumCircuit:
        qreg = QuantumRegister(1)
        new_circ = QuantumCircuit(qreg, *circ.cregs)
        qubit = circ.qubits[qubit_index]
        for instr in circ:
            if tuple(instr.qubits) == (qubit,):
                new_circ.append(instr.operation, [qreg[0]], instr.clbits)
        return new_circ


class WireCut(Gate):
    def __init__(
        self,
        label: str | None = None,
    ) -> None:
        super().__init__(name="wire_cut", num_qubits=1, params=[], label=label)


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
