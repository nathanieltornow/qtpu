import numpy as np
from numpy import ndarray
from qiskit.circuit import QuantumCircuit

from qvm.instructions import VirtualBinaryGate


class VirtualRZZ(VirtualBinaryGate):
    def instantiations(self) -> list[tuple[QuantumCircuit, QuantumCircuit]]:
        z = QuantumCircuit(1, 1)
        z.z(0)

        s = QuantumCircuit(1, 1)
        s.s(0)

        sdg = QuantumCircuit(1, 1)
        sdg.sdg(0)

        meas = QuantumCircuit(1, 1)
        meas.measure(0, 0)

        i = QuantumCircuit(1, 1)

        return [
            (i, i),
            (z, z),
            (meas, s),
            (meas, sdg),
            (s, meas),
            (sdg, meas),
        ]

    def coefficients_1d(self) -> np.ndarray:
        theta = -self.original_gate.params[0] / 2
        cs = np.cos(theta) * np.sin(theta)
        return np.array([np.cos(theta) ** 2, np.sin(theta) ** 2, -cs, cs, -cs, cs])

    def instantiations_qubit0(self) -> list[QuantumCircuit]:
        z = QuantumCircuit(1, 1)
        z.z(0)

        s = QuantumCircuit(1, 1)
        s.s(0)

        sdg = QuantumCircuit(1, 1)
        sdg.sdg(0)

        meas = QuantumCircuit(1, 1)
        meas.measure(0, 0)

        i = QuantumCircuit(1, 1)

        return [i, z, meas, s, sdg]

    def instantiations_qubit1(self) -> list[QuantumCircuit]:
        return self.instantiations_qubit0()

    def coefficients_2d(self) -> ndarray:
        c = self.coefficients_1d()
        return np.array(
            [
                [c[0], 0, 0, 0, 0],
                [0, c[1], 0, 0, 0],
                [0, 0, 0, c[2], c[3]],
                [0, 0, c[4], 0, 0],
                [0, 0, c[5], 0, 0],
            ],
            dtype=np.float64,
        )
