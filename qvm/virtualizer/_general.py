from multiprocessing.pool import Pool
from qiskit.circuit import QuantumCircuit
from qvm.quasi_distr import QuasiDistr
from qvm.types import Argument, Fragment

from qvm.util import fragment_circuit
from qvm.virtual_gates import VirtualSWAP, VirtualBinaryGate
from ._virtualizer import Virtualizer
from ._gate_virtualizer import OneFragmentGateVirtualizer
from ._wire_virtualizer import SingleWireVirtualizer


class TwoFragmentVirtualizer(Virtualizer):
    """A virtualizer which can virtualize a single wire and arbitrary number of gates.

    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        circuit = fragment_circuit(circuit)
        # right now, we only support one wire cut
        if (
            not sum(1 for instr in circuit if isinstance(instr.operation, VirtualSWAP))
            <= 1
        ):
            raise ValueError("Circuit must have exactly one wire cut.")
        if len(circuit.qregs) != 2:
            raise ValueError("Circuit must have exactly two fragments")
        super().__init__(circuit)
        self._vgates = [instr.operation for instr in circuit if isinstance(instr.operation, VirtualBinaryGate)]

    def instantiate(self) -> dict[Fragment, list[Argument]]:
        raise NotImplementedError

    def knit(self, results: dict[Fragment, list[QuasiDistr]], pool: Pool) -> QuasiDistr:
        raise NotImplementedError
