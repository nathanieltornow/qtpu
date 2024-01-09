import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit

from qvm.instructions import VirtualBinaryGate


class VirtualMove(VirtualBinaryGate):
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
