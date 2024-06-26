from qiskit.circuit import QuantumCircuit

from .cz import VirtualCZ


class VirtualCX(VirtualCZ):
    def __init__(self, label: str | None = None) -> None:
        super().__init__(label)
        self._name = "v_cx"

    def instantiations(self) -> list[QuantumCircuit]:
        h = QuantumCircuit(1)
        h.h(0)

        cx_insts = []
        for cz_inst in super().instantiations():
            cx_inst = cz_inst.copy()
            cx_inst.compose(h, qubits=[1], inplace=True, front=True)
            cx_inst.compose(h, qubits=[1], inplace=True, front=False)
            cx_insts.append(cx_inst)
        return cx_insts

    def instances_q1(self) -> list[QuantumCircuit]:
        h = QuantumCircuit(1, 1)
        h.h(0)

        cx_insts = []
        for inst in super().instances_q1():
            new_inst = inst.copy()
            new_inst.compose(h, inplace=True, front=True)
            new_inst.compose(h, inplace=True, front=False)
            cx_insts.append(new_inst)
        return cx_insts
