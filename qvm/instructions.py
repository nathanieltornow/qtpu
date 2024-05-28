import abc

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister, Barrier


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


class InstanceGate(Barrier):
    def __init__(
        self,
        num_qubits: int,
        index: str,
        instances: list[QuantumCircuit],
        shot_portion: list[float] | None = None,
    ):
        assert all(inst.num_qubits == num_qubits for inst in instances)

        if shot_portion is None:
            shot_portion = [1 / len(instances) for _ in instances]

        # sum of shot_portion must be approximately 1
        assert abs(sum(shot_portion) - 1) < 1e-6

        self._index = index
        self._instances = instances
        self._shot_portion = shot_portion
        super().__init__(num_qubits, label=index)

    @property
    def index(self) -> str:
        return self._index

    @property
    def instances(self) -> list[QuantumCircuit]:
        return self._instances

    @property
    def shot_portion(self) -> list[float]:
        return self._shot_portion
