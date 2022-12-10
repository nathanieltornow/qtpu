from typing import List

from qiskit.circuit.quantumcircuit import QuantumCircuit

from vqc.prob_distr import ProbDistr
from vqc.types import VirtualGate


class VirtualIdentityGate(VirtualGate):
    def configure(self) -> List[QuantumCircuit]:
        return [QuantumCircuit(2)]

    def knit(self, results: List[ProbDistr]) -> ProbDistr:
        return results[0].without_first_bit()[0]
