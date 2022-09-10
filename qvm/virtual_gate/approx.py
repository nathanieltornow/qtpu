from math import pi, sin, cos
from typing import List, Type

from qiskit.circuit.quantumcircuit import QuantumCircuit, Instruction
from qiskit.circuit.library.standard_gates import CZGate, CXGate, RZZGate

from qvm.prob import ProbDistribution
from qvm.virtual_gate.virtual_gate import VirtualBinaryGate


class NoVirtualGate(VirtualBinaryGate):
    pass


class ApproxVirtualCZ(VirtualBinaryGate):
    def configure(self) -> List[QuantumCircuit]:
        conf0 = QuantumCircuit(2, 1)
        conf0.rz(pi / 2, 0)
        conf0.rz(pi / 2, 1)

        conf1 = QuantumCircuit(2, 1)
        conf1.rz(-pi / 2, 0)
        conf1.rz(-pi / 2, 1)

        return [conf0, conf1]

    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        r0, _ = results[0].without_first_bit()
        r1, _ = results[1].without_first_bit()
        return (r0 + r1) * 0.5


class ApproxVirtualCX(VirtualBinaryGate):
    def configure(self) -> List[QuantumCircuit]:
        conf0 = QuantumCircuit(2, 1)
        conf0.rz(pi / 2, 0)
        conf0.h(1)
        conf0.rz(pi / 2, 1)
        conf0.h(1)

        conf1 = QuantumCircuit(2, 1)
        conf1.rz(-pi / 2, 0)
        conf1.h(1)
        conf1.rz(-pi / 2, 1)
        conf1.h(1)

        return [conf0, conf1]

    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        r0, _ = results[0].without_first_bit()
        r1, _ = results[1].without_first_bit()
        return (r0 + r1) * 0.5


class ApproxVirtualRZZ(VirtualBinaryGate):
    def configure(self) -> List[QuantumCircuit]:
        conf0 = QuantumCircuit(2, 1)

        conf1 = QuantumCircuit(2, 1)
        conf1.z(0)
        conf1.z(1)

        return [conf0, conf1]

    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        r0, _ = results[0].without_first_bit()
        r1, _ = results[1].without_first_bit()
        theta = -self.params[0]
        return (r0 * cos(theta / 2) ** 2) + (r1 * sin(theta / 2) ** 2)
