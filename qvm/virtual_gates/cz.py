import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit

from qvm.instructions import VirtualBinaryGate


class VirtualCZ(VirtualBinaryGate):
    def __init__(self, label: str | None = None) -> None:
        super().__init__(name="v_cz", params=[], label=label)

    def instantiations(self) -> list[QuantumCircuit]:
        c1 = QuantumCircuit(2, 1)
        c1.sdg(0)
        c1.sdg(1)

        c2 = QuantumCircuit(2, 1)
        c2.s(0)
        c2.s(1)

        c3 = QuantumCircuit(2, 1)
        c3.sdg(0)
        c3.measure(0, 0)

        c4 = QuantumCircuit(2, 1)
        c4.sdg(0)
        c4.measure(0, 0)
        c4.z(1)

        c5 = QuantumCircuit(2, 1)
        c5.sdg(1)
        c5.measure(1, 0)

        c6 = QuantumCircuit(2, 1)
        c6.z(0)
        c6.sdg(1)
        c6.measure(1, 0)

        return [c1, c2, c3, c4, c5, c6]

    def coefficients_1d(self) -> NDArray[np.float32]:
        return 0.5 * np.array([1, 1, 1, -1, 1, -1], dtype=np.float32)

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

    def coefficients_2d(self) -> NDArray[np.float32]:
        return 0.5 * np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, -1],
                [0, 0, 1, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float32,
        )
