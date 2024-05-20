import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit

from .rzz import VirtualRZZ


class VirtualCPhase(VirtualRZZ):
    def __init__(self, params: list, label: str | None = None) -> None:
        super().__init__("v_rzz", params, label)
        self._params[0] = -self._params[0] / 2

    def instantiations(self) -> list[QuantumCircuit]:
        lam = self.params[0] / 2
        assert isinstance(lam, float)

        lam_circ1 = QuantumCircuit(2, 1)
        lam_circ1.rz(lam, 0)

        lam_circ2 = QuantumCircuit(2, 1)

        insts = []

        for rzz_inst in super().instantiations():
            insts.append(lam_circ1.compose(rzz_inst).compose(lam_circ2))

        return insts

    def instances_q0(self) -> list[QuantumCircuit]:
        lam = self.params[0] / 2

        lam_circ = QuantumCircuit(1, 1)
        lam_circ.rz(lam, 0)

        insts = []

        for rzz_inst in super().instantiations():
            insts.append(lam_circ.compose(rzz_inst))

        return insts

    def instances_q1(self) -> list[QuantumCircuit]:
        lam = self.params[0] / 2

        lam_circ = QuantumCircuit(1, 1)
        lam_circ.rz(lam, 0)

        insts = []

        for rzz_inst in super().instantiations():
            insts.append(rzz_inst.compose(lam_circ))

        return insts
