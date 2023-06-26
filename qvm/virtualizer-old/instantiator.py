from typing import Iterable
from math import pi

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library import RZZGate, CZGate, RZGate, Measure
from qvm.types import Argument


CZ_0_INSTANTIATIONS = [
    RZGate(pi / 2),
    RZGate(-pi / 2),
    RZZGate(pi),
    Measure(),
    Measure(),
    RZGate(0.0),
]

CZ_1_INSTANTIATIONS = [
    RZGate(pi / 2),
    RZGate(-pi / 2),
    Measure(),
    RZZGate(pi),
    RZGate(0.0),
    Measure(),
]

RZZ_0_INSTANTIATIONS = [
    RZGate(0.0),
]

RZZ_1_INSTANTIATIONS = [
    RZGate(0.0),
]


# def _inst_ids(num_vgates: int) -> Ite

# def instantiate(
#     vgates: list[RZZGate | CZGate],
# ) -> list[Argument]:
    
#     pass


