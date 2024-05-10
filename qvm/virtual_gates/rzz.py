import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit

from qvm.instructions import VirtualBinaryGate


class VirtualRZZ(VirtualBinaryGate):
    def __init__(self, params: list, label: str | None = None) -> None:
        super().__init__("v_rzz", params, label)

    def instantiations(self) -> list[tuple[QuantumCircuit, QuantumCircuit]]:
        c1 = QuantumCircuit(2, 1)

        c2 = QuantumCircuit(2, 1)
        c2.z(0)
        c2.z(1)

        c3 = QuantumCircuit(2, 1)
        c3.measure(0, 0)
        c3.s(1)

        c4 = QuantumCircuit(2, 1)
        c4.measure(0, 0)
        c4.sdg(1)

        c5 = QuantumCircuit(2, 1)
        c5.s(0)
        c5.measure(1, 0)

        c6 = QuantumCircuit(2, 1)
        c6.sdg(0)
        c6.measure(1, 0)

        return [c1, c2, c3, c4, c5, c6]

    def coefficients_1d(self) -> NDArray[np.float32]:
        theta = -self.params[0] / 2
        cs = np.cos(theta) * np.sin(theta)
        return np.array(
            [np.cos(theta) ** 2, np.sin(theta) ** 2, -cs, cs, -cs, cs], dtype=np.float32
        )

    def instances_q0(self) -> list[QuantumCircuit]:
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

    def instances_q1(self) -> list[QuantumCircuit]:
        return self.instances_q0()

    def coefficients_2d(self) -> NDArray[np.float32]:
        c = self.coefficients_1d()
        return np.array(
            [
                [c[0], 0, 0, 0, 0],
                [0, c[1], 0, 0, 0],
                [0, 0, 0, c[2], c[3]],
                [0, 0, c[4], 0, 0],
                [0, 0, c[5], 0, 0],
            ],
            dtype=np.float32,
        )
