import numpy as np
from numpy import ndarray
from qiskit.circuit import QuantumCircuit

from qvm.instructions import VirtualBinaryGate


class VirtualCZ(VirtualBinaryGate):
    def instantiations(self) -> list[tuple[QuantumCircuit, QuantumCircuit]]:
        sdg = QuantumCircuit(1, 1)
        sdg.sdg(0)

        s = QuantumCircuit(1, 1)
        s.s(0)

        z = QuantumCircuit(1, 1)
        z.z(0)

        i = QuantumCircuit(1, 1)

        sdg_meas = QuantumCircuit(1, 1)
        sdg_meas.sdg(0)
        sdg_meas.measure(0, 0)

        return [
            (sdg, sdg),
            (s, s),
            (sdg_meas, i),
            (sdg_meas, z),
            (i, sdg_meas),
            (z, sdg_meas),
        ]

    def coefficients_1d(self) -> np.ndarray:
        return 0.5 * np.array([1, 1, 1, -1, 1, -1])

    def instantiations_qubit0(self) -> list[QuantumCircuit]:
        sdg = QuantumCircuit(1, 1)
        sdg.sdg(0)

        s = QuantumCircuit(1, 1)
        s.s(0)

        z = QuantumCircuit(1, 1)
        z.z(0)

        i = QuantumCircuit(1, 1)

        sdg_meas = QuantumCircuit(1, 1)
        sdg_meas.sdg(0)
        sdg_meas.measure(0, 0)

        return [sdg, s, sdg_meas, i, z]

    def instantiations_qubit1(self) -> list[QuantumCircuit]:
        return self.instantiations_qubit0()

    def coefficients_2d(self) -> ndarray:
        return 0.5 * np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, -1],
                [0, 0, 1, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float64,
        )
