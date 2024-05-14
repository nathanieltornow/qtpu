import abc

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import (
    Gate,
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

    @abc.abstractmethod
    def instantiations(self) -> list[QuantumCircuit]: ...

    @abc.abstractmethod
    def coefficients_1d(self) -> NDArray[np.float32]:
        pass

    def instances_q0(self) -> list[QuantumCircuit]:
        return [self._circ_on_qubit(inst, 0) for inst in self.instantiations()]

    def instances_q1(self) -> list[QuantumCircuit]:
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
