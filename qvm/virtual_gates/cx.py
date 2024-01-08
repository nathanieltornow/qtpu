from qiskit.circuit import QuantumCircuit

from .cz import VirtualCZ


class VirtualCX(VirtualCZ):
    def instantiations(self) -> list[tuple[QuantumCircuit, QuantumCircuit]]:
        insts = super().instantiations()
        h = QuantumCircuit(1, 1)
        h.h(0)

        cx_insts = []
        for inst0, inst1 in insts:
            new_inst = inst1.copy()
            new_inst.compose(h, inplace=True, front=True)
            new_inst.compose(h, inplace=True, front=False)
            cx_insts.append((inst0, new_inst))
        return cx_insts

    def instantiations_qubit1(self) -> list[QuantumCircuit]:
        h = QuantumCircuit(1, 1)
        h.h(0)

        cx_insts = []
        for inst in super().instantiations_qubit1():
            new_inst = inst.copy()
            new_inst.compose(h, inplace=True, front=True)
            new_inst.compose(h, inplace=True, front=False)
            cx_insts.append(new_inst)
        return cx_insts
