import abc
from math import pi, cos, sin

from qiskit.circuit import QuantumCircuit, Barrier, Gate

from qvm.quasi_distr import QuasiDistr


class WireCut(Barrier):
    def __init__(self):
        super().__init__(num_qubits=1, label="wire_cut")


class VirtualQubitChannel(Barrier):
    def __init__(self):
        super().__init__(num_qubits=2, label="v_chan")


class VirtualBinaryGate(Barrier, abc.ABC):
    def __init__(self, original_gate: Gate):
        self._original_gate = original_gate
        super().__init__(num_qubits=original_gate.num_qubits, label=original_gate.name)
        for inst in self._instantiations():
            self._check_instantiation(inst)
        self._name = f"v_{original_gate.name}"

    @abc.abstractmethod
    def _instantiations(self) -> list[QuantumCircuit]:
        pass

    @abc.abstractmethod
    def knit(self, results: list[QuasiDistr]) -> QuasiDistr:
        pass

    def instantiate(self, inst_id: int) -> QuantumCircuit:
        return self._instantiations()[inst_id]

    def _check_instantiation(self, inst: QuantumCircuit):
        assert len(inst.qubits) == 2
        assert len(inst.clbits) == 1
        for instr in inst.data:
            assert len(instr.qubits) == 1
            assert len(instr.clbits) <= 1

    def _define(self):
        circuit = QuantumCircuit(2)
        circuit.append(self._original_gate, [0, 1], [])
        self._definition = circuit


class VirtualCZ(VirtualBinaryGate):
    def _instantiations(self) -> list[QuantumCircuit]:
        inst0 = QuantumCircuit(2, 1)
        inst0.rz(pi / 2, 0)
        inst0.rz(pi / 2, 1)

        inst1 = QuantumCircuit(2, 1)
        inst1.rz(-pi / 2, 0)
        inst1.rz(-pi / 2, 1)

        inst2 = QuantumCircuit(2, 1)
        inst2.rz(pi, 0)
        inst2.measure(1, 0)

        inst3 = QuantumCircuit(2, 1)
        inst3.measure(0, 0)
        inst3.rz(pi, 1)

        inst4 = QuantumCircuit(2, 1)
        inst4.measure(0, 0)

        inst5 = QuantumCircuit(2, 1)
        inst5.measure(1, 0)

        return [inst0, inst1, inst2, inst3, inst4, inst5]

    def knit(self, results: list[QuasiDistr]) -> QuasiDistr:
        r0, _ = results[0].divide_by_first_bit()
        r1, _ = results[1].divide_by_first_bit()
        r20, r21 = results[2].divide_by_first_bit()
        r30, r31 = results[3].divide_by_first_bit()
        r40, r41 = results[4].divide_by_first_bit()
        r50, r51 = results[5].divide_by_first_bit()
        return (r0 + r1 + (r21 - r20) + (r31 - r30) + (r40 - r41) + (r50 - r51)) * 0.5


class VirtualCX(VirtualCZ):
    def _instantiations(self) -> list[QuantumCircuit]:
        h_gate_circ = QuantumCircuit(2, 1)
        h_gate_circ.h(1)

        cz_insts = []
        for inst in super()._instantiations():
            new_inst = h_gate_circ.compose(inst, inplace=False)
            cz_insts.append(new_inst.compose(h_gate_circ, inplace=False))
        return cz_insts


class VirtualCY(VirtualCX):
    def _instantiations(self) -> list[QuantumCircuit]:
        minus_rz = QuantumCircuit(2, 1)
        minus_rz.rz(-pi / 2, 1)
        plus_rz = QuantumCircuit(2, 1)
        plus_rz.rz(pi / 2, 1)

        cy_insts = []
        for inst in super()._instantiations():
            new_inst = minus_rz.compose(inst, inplace=False)
            cy_insts.append(new_inst.compose(plus_rz, inplace=False))
        return cy_insts


class VirtualRZZ(VirtualBinaryGate):
    def _instantiations(self) -> list[QuantumCircuit]:
        inst0 = QuantumCircuit(2, 1)

        inst1 = QuantumCircuit(2, 1)
        inst1.z(0)
        inst1.z(1)

        inst2 = QuantumCircuit(2, 1)
        inst2.rz(-pi / 2, 0)
        inst2.measure(1, 0)

        inst3 = QuantumCircuit(2, 1)
        inst3.measure(0, 0)
        inst3.rz(-pi / 2, 1)

        inst4 = QuantumCircuit(2, 1)
        inst4.rz(pi / 2, 0)
        inst4.measure(1, 0)

        inst5 = QuantumCircuit(2, 1)
        inst5.measure(0, 0)
        inst5.rz(pi / 2, 1)

        return [inst0, inst1, inst2, inst3, inst4, inst5]

    def knit(self, results: list[QuasiDistr]) -> QuasiDistr:
        r0, _ = results[0].divide_by_first_bit()
        r1, _ = results[1].divide_by_first_bit()
        r23 = results[2] + results[3]
        r45 = results[4] + results[5]

        r230, r231 = r23.divide_by_first_bit()
        r450, r451 = r45.divide_by_first_bit()

        theta = -self.params[0]
        return (
            (r0 * cos(theta / 2) ** 2)
            + (r1 * sin(theta / 2) ** 2)
            + (r230 - r231 - r450 + r451) * cos(theta / 2) * sin(theta / 2)
        )
