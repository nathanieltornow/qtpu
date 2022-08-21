from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, final

from qiskit import transpile
from qiskit.circuit.quantumcircuit import QuantumCircuit, Qubit
from qiskit.providers import Backend

from qvm.circuit import (
    VirtualCircuitBase,
    FragmentedVirtualCircuit,
    VirtualCircuit,
    Fragment,
)
from qvm.circuit.virtual_gate.virtual_gate import VirtualBinaryGate


class TranspiledVirtualCircuit(VirtualCircuit, ABC):
    _backend: Backend
    _transpile_kwargs: Dict[str, Any] = {}

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Backend,
    ) -> None:
        super().__init__(circuit)
        self._backend = backend
        self._transpile_kwargs = self.transpile_kwargs()

    @abstractmethod
    def transpile_kwargs(self) -> Dict[str, Any]:
        pass

    def configured_circuits(self) -> Iterator[Tuple[Tuple[int, ...], QuantumCircuit]]:
        conf_circs = super().configured_circuits()
        for conf, circ in conf_circs:
            t_circ = transpile(circ, backend=self._backend, **self.transpile_kwargs)
            yield conf, t_circ


class TranspiledFragment(Fragment):
    _tvc: Type[TranspiledVirtualCircuit]
    _backend: Backend

    def __init__(
        self,
        original_circuit: VirtualCircuitBase,
        qubits: Set[Qubit],
        transpiled_virtual_circuit_t: Type[TranspiledVirtualCircuit],
        backend: Backend,
    ) -> None:
        super().__init__(original_circuit, qubits)
        self._tvc = transpiled_virtual_circuit_t
        self._backend = backend

    def configured_circuits(
        self, virtual_gates: List[VirtualBinaryGate]
    ) -> Iterator[Tuple[Tuple[int, ...], VirtualCircuit]]:
        conf_circs = super().configured_circuits(virtual_gates)
        for conf_id, circ in conf_circs:
            yield conf_id, self._tvc(circ, self._backend)


class TranspiledFragmentedCircuit(FragmentedVirtualCircuit):
    def transpile(
        self,
        fragment: Fragment,
        transpiled_virtual_circuit_t: Type[TranspiledVirtualCircuit],
        backend: Backend,
    ) -> None:
        self._fragments.remove(fragment)
        t_frag = TranspiledFragment(
            self, fragment.qubit_indices(), transpiled_virtual_circuit_t, backend
        )
        self._fragments.add(t_frag)
