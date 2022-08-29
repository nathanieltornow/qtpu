from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from qiskit.circuit import QuantumCircuit, CircuitInstruction, Qubit
from qiskit.providers import Backend

from qvm.virtual_gate import VirtualBinaryGate, VirtualCZ, VirtualCX, VirtualRZZ
from .fragmented_circuit import FragmentedCircuit

STANDARD_VIRTUAL_GATES: Dict[str, Type[VirtualBinaryGate]] = {
    "cz": VirtualCZ,
    "cx": VirtualCX,
    "rzz": VirtualRZZ,
}


class DecompositionTranspiler(ABC):
    @abstractmethod
    def run(self, circuit: QuantumCircuit) -> FragmentedCircuit:
        pass


class LayoutTranspiler(ABC):
    @abstractmethod
    def run(
        self, circuit: QuantumCircuit, backend: Optional[Backend] = None
    ) -> QuantumCircuit:
        pass


class MappingTranspiler(ABC):
    @abstractmethod
    def run(
        self, frag_circ: FragmentedCircuit, available_backends: List[Backend]
    ) -> None:
        pass


def virtualize_connection(
    circuit: QuantumCircuit,
    qubit1: Qubit,
    qubit2: Qubit,
    virtual_gates: Dict[str, Type[VirtualBinaryGate]] = STANDARD_VIRTUAL_GATES,
):
    if not {qubit1, qubit2} <= set(circuit.qubits):
        raise ValueError(f"qubits {qubit1, qubit2} not in circuit")
    for i in range(len(circuit.data)):
        circ_instr = circuit.data[i]
        if set(circ_instr.qubits) == {qubit1, qubit2} and len(circ_instr.clbits) == 0:
            vgate = virtual_gates[circ_instr.operation.name](
                circ_instr.operation.params
            )
            circuit.data[i] = CircuitInstruction(vgate, circ_instr.qubits, ())
