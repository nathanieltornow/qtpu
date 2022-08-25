from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, final

from qiskit import transpile
from qiskit.circuit.quantumcircuit import QuantumCircuit, Qubit
from qiskit.providers import Backend

from qvm.circuit import (
    Fragment,
)
from qvm.circuit.virtual_circuit import VirtualCircuit, VirtualCircuitInterface
from qvm.circuit.virtual_gate.virtual_gate import VirtualBinaryGate

DEFAULT_TRANSPILER_FLAGS = {"optimization_level": 3}
DEFAULT_EXEC_FLAGS = {"shots": 10000}


class TranspiledFragment(Fragment):
    backend: Backend
    transpile_flags: Dict[str, Any]
    exec_flags: Dict[str, Any]

    def __init__(
        self,
        virtual_circuit: VirtualCircuitInterface,
        qubits: Set[Qubit],
        backend: Backend,
        transpile_flags: Dict[str, Any] = DEFAULT_TRANSPILER_FLAGS,
        exec_flags: Dict[str, Any] = DEFAULT_EXEC_FLAGS,
    ) -> None:
        super().__init__(virtual_circuit, qubits)
        self.backend = backend
        self.transpile_flags = transpile_flags
        self.exec_flags = exec_flags


class TranspiledVirtualCircuit(VirtualCircuit):
    @staticmethod
    def from_virtual_circuit(vc: VirtualCircuit) -> "TranspiledVirtualCircuit":
        vc = TranspiledVirtualCircuit(
            *vc.qregs.copy(),
            *vc.cregs.copy(),
            name=vc.name,
            global_phase=vc.global_phase,
            metadata=vc.metadata,
        )
        for circ_instr in vc.data:
            vc.append(circ_instr.copy())
        return vc

    def transpile_fragment(
        self,
        fragment: Fragment,
        backend: Backend,
        transpile_flags: Dict[str, Any] = DEFAULT_TRANSPILER_FLAGS,
        exec_flags: Dict[str, Any] = DEFAULT_EXEC_FLAGS,
    ) -> None:
        if fragment not in self._fragments:
            raise ValueError(f"Fragment {fragment} not in virtual circuit")
        self._fragments.remove(fragment)
        self._fragments.add(
            TranspiledFragment(
                self, fragment.qubits, backend, transpile_flags, exec_flags
            )
        )
