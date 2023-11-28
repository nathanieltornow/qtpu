import abc

import numpy as np
from qiskit.circuit import (
    Barrier,
    Gate,
    Instruction,
    Measure,
    QuantumCircuit,
)
from qiskit.circuit.library.standard_gates import (
    HGate,
    RZGate,
    SdgGate,
    SGate,
    SXdgGate,
    SXGate,
    XGate,
    ZGate,
)

from qvm.quasi_distr import QuasiDistr


class WireCut(Barrier):
    def __init__(self):
        super().__init__(num_qubits=1, label="wc")

    def _define(self):
        self._definition = QuantumCircuit(1)


class VirtualBinaryGate(Barrier, abc.ABC):
    def __init__(self, original_gate: Gate):
        self._original_gate = original_gate
        super().__init__(num_qubits=original_gate.num_qubits, label=original_gate.name)
        self._name = f"v_{original_gate.name}"
        self._params = original_gate.params

    @property
    def original_gate(self) -> Gate:
        return self._original_gate

    @property
    def num_instantiations(self) -> int:
        return len(self._instantiations())

    @abc.abstractmethod
    def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
        pass

    @abc.abstractmethod
    def coefficients(self) -> np.ndarray:
        pass

    def get_coefficient(self, inst_id: int) -> float:
        return self.coefficients()[inst_id]

    def _define(self):
        circuit = QuantumCircuit(2)
        circuit.append(self._original_gate, [0, 1], [])
        self._definition = circuit


class VirtualGateEndpoint(Barrier):
    def __init__(self, virtual_gate: VirtualBinaryGate, vgate_idx: int, qubit_idx: int):
        self._virtual_gate = virtual_gate
        self.vgate_idx = vgate_idx
        self.qubit_idx = qubit_idx
        super().__init__(
            num_qubits=1, label=f"{virtual_gate.name}_{vgate_idx}_{qubit_idx}"
        )

    def instantiate(self, inst_id: int) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        gates = self._virtual_gate._instantiations()[inst_id][self.qubit_idx]
        for gate in gates:
            circuit.append(gate, [0], [0] * gate.num_clbits)
        return circuit


class VirtualCZ(VirtualBinaryGate):
    def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
        return [
            (
                [SdgGate()],
                [SdgGate()],
            ),
            (
                [SGate()],
                [SGate()],
            ),
            (
                [SdgGate(), Measure()],
                [],
            ),
            (
                [SdgGate(), Measure()],
                [ZGate()],
            ),
            (
                [],
                [SdgGate(), Measure()],
            ),
            (
                [ZGate()],
                [SdgGate(), Measure()],
            ),
        ]

    def coefficients(self) -> np.ndarray:
        return 0.5 * np.array([1, 1, 1, -1, 1, -1])


class VirtualCX(VirtualCZ):
    def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
        cx_insts = []
        for cz_inst in super()._instantiations():
            cx_insts.append((cz_inst[0], [HGate()] + cz_inst[1] + [HGate()]))
        return cx_insts


class VirtualCY(VirtualCX):
    def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
        cy_insts = []
        for cx_inst in super()._instantiations():
            cy_insts.append(
                (
                    cx_inst[0],
                    [RZGate(-np.pi / 2)] + cx_inst[1] + [RZGate(np.pi / 2)],
                )
            )
        return cy_insts


class VirtualRZZ(VirtualBinaryGate):
    def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
        pauli = ZGate()
        r_plus = SGate()
        r_minus = SdgGate()
        meas = Measure()
        return [
            ([], []),  # Identity
            ([pauli], [pauli]),
            ([meas], [r_plus]),
            ([meas], [r_minus]),
            ([r_plus], [meas]),
            ([r_minus], [meas]),
        ]

    def coefficients(self) -> np.ndarray:
        theta = -self._params[0] / 2
        cs = np.cos(theta) * np.sin(theta)
        return np.array(
            [
                np.cos(theta) ** 2,
                np.sin(theta) ** 2,
                -cs,
                cs,
                -cs,
                cs,
            ]
        )


class VirtualMove(VirtualBinaryGate):
    def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
        return [
            ([], []),
            (
                [],
                [XGate()],
            ),
            (
                [HGate(), Measure()],
                [HGate()],
            ),
            (
                [HGate(), Measure()],
                [XGate(), HGate()],
            ),
            (
                [SXGate(), Measure()],
                [SXdgGate()],
            ),
            (
                [SXGate(), Measure()],
                [XGate(), SXdgGate()],
            ),
            (
                [Measure()],
                [],
            ),
            (
                [Measure()],
                [XGate()],
            ),
        ]

    def coefficients(self) -> np.ndarray:
        return 0.5 * np.array([1, 1, 1, -1, 1, -1, 1, -1])


VIRTUAL_GATE_TYPES = {
    "cx": VirtualCX,
    "cy": VirtualCY,
    "cz": VirtualCZ,
    "rzz": VirtualRZZ,
}
