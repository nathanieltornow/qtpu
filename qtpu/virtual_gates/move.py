import numpy as np
from numpy import float32
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit

from qtpu.instructions import VirtualBinaryGate


class VirtualMove(VirtualBinaryGate):
    def __init__(self, label: str | None = None) -> None:
        super().__init__(name="v_move", params=[], label=label)

    def instantiations(self) -> list[QuantumCircuit]:
        c1 = QuantumCircuit(2, 1)

        c2 = QuantumCircuit(2, 1)
        c2.x(1)

        c3 = QuantumCircuit(2, 1)
        c3.h(0)
        c3.measure(0, 0)
        c3.h(1)

        c4 = QuantumCircuit(2, 1)
        c4.h(0)
        c4.measure(0, 0)
        c4.x(1)
        c4.h(1)

        c5 = QuantumCircuit(2, 1)
        c5.sx(0)
        c5.measure(0, 0)
        c5.sxdg(1)

        c6 = QuantumCircuit(2, 1)
        c6.sx(0)
        c6.measure(0, 0)
        c6.x(1)
        c6.sxdg(1)

        c7 = QuantumCircuit(2, 1)
        c7.measure(0, 0)

        c8 = QuantumCircuit(2, 1)
        c8.measure(0, 0)
        c8.x(1)

        return [c1, c2, c3, c4, c5, c6, c7, c8]

    def coefficients_1d(self) -> NDArray[np.float32]:
        return 0.5 * np.array([1, 1, 1, -1, 1, -1, 1, -1], dtype=np.float32)

    def instances_q0(self) -> list[QuantumCircuit]:
        i = QuantumCircuit(1, 1)
        i.reset(0)

        z = QuantumCircuit(1, 1)
        z.measure(0, 0)
        z.reset(0)

        x = QuantumCircuit(1, 1)
        x.h(0)
        x.measure(0, 0)
        x.reset(0)

        y = QuantumCircuit(1, 1)
        y.sx(0)
        y.measure(0, 0)
        y.reset(0)

        return [i, z, x, y]

    def instances_q1(self) -> list[QuantumCircuit]:
        zero = QuantumCircuit(1, 1)
        zero.reset(0)

        one = QuantumCircuit(1, 1)
        one.reset(0)
        one.x(0)

        plus = QuantumCircuit(1, 1)
        plus.reset(0)
        plus.h(0)

        iplus = QuantumCircuit(1, 1)
        iplus.reset(0)
        iplus.sxdg(0)

        return [zero, one, plus, iplus]

    def coefficients_2d(self) -> NDArray[float32]:
        A = np.array(
            [[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        B = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [-1, -1, 2, 0], [-1, -1, 0, 2]],
            dtype=np.float32,
        )
        return 0.5 * A @ B



if __name__ == "__main__":
    print(VirtualMove().coefficients_2d())