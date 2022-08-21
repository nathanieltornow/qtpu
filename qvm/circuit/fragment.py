import itertools
from typing import Dict, Iterator, List, Set, Tuple, Type

import networkx as nx
from qiskit.circuit.quantumcircuit import (
    QuantumCircuit,
    Qubit,
    Instruction,
    CircuitInstruction,
)

from .virtual_gate import VirtualBinaryGate, PartialVirtualGate
from .virtual_circuit import STANDARD_VIRTUAL_GATES, VirtualCircuitBase, VirtualCircuit
from qvm import util


class Fragment(VirtualCircuitBase):
    _original_circuit: VirtualCircuit

    _qubit_indices: Set[int]

    def __init__(
        self, original_circuit: VirtualCircuit, qubit_indices: Set[int]
    ) -> None:
        self._original_circuit = original_circuit
        self._qubit_indices = qubit_indices

    def __hash__(self) -> int:
        return min(self._qubit_indices)

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    def qubit_indices(self) -> Set[int]:
        return self._qubit_indices.copy()

    def _instr_map(self) -> Dict[int, int]:
        orig = self._original_circuit.circuit()
        instr_map = {}
        frag_index = 0
        for instr_index, instr in enumerate(orig.data):
            if set(util.bit_indices(orig, instr.qubits)) <= self._qubit_indices:
                instr_map[frag_index] = instr_index
                frag_index += 1
        return instr_map

    def circuit(self) -> QuantumCircuit:
        orig = self._original_circuit.circuit()
        return util.circuit_on_qubits(
            orig, {orig.qubits[i] for i in self._qubit_indices}
        )

    def virtualize_gate(
        self, instr_index: int, virtual_gate_t: Type[VirtualBinaryGate]
    ) -> None:
        instr_map = self._instr_map()
        if instr_index not in instr_map:
            raise ValueError("instruction not in fragment")
        self._original_circuit.virtualize_gate(instr_map[instr_index], virtual_gate_t)

    def virtualize_connection(
        self,
        qubit1_index: int,
        qubit2_index: int,
        virtual_gates: Dict[str, Type[VirtualBinaryGate]],
    ) -> None:
        og_circ = self._original_circuit.circuit()
        if not {qubit1_index, qubit2_index} <= self._qubit_indices:
            raise ValueError("qubits not in fragment")
        qubit1 = og_circ.qubits[qubit1_index]
        qubit2 = og_circ.qubits[qubit2_index]
        for i, instr in enumerate(og_circ.data):
            if {qubit1, qubit2}.issubset(instr.qubits):
                self._original_circuit.virtualize_gate(
                    i, virtual_gates[instr.operation.name]
                )


class FragmentedVirtualCircuit(VirtualCircuit):
    _fragments: Set[Fragment]

    def __init__(self, circuit: QuantumCircuit) -> None:
        super().__init__(circuit)
        self._fragments = {Fragment(self, set(range(circuit.num_qubits)))}

    def fragments(self) -> Set[Fragment]:
        return self._fragments.copy()

    def create_fragments(self) -> None:
        self._fragments = set()
        node_sets = nx.connected_components(self.connectivity_graph())
        for node_set in list(node_sets):
            self._fragments.add(Fragment(self, node_set))
        print(self._fragments)

    def merge_fragments(self, fragments: Set[Fragment]) -> None:
        merged_qubits: Set[Qubit] = set()
        for frag in fragments:
            self._fragments.remove(frag)
            merged_qubits |= frag.qubit_indices()
        self._fragments.add(Fragment(self, merged_qubits))

    def fragment_virtual_gates(self) -> List[VirtualBinaryGate]:
        res_virtual_gates: List[VirtualBinaryGate] = []
        for instr in self._circuit.data:
            # if the virtual gate is not entirely in one fragment, it has to be
            # handeled by the original circuit
            if isinstance(instr.operation, VirtualBinaryGate) and not any(
                set(instr.qubits).issubset(frag.qubit_indices())
                for frag in self._fragments
            ):
                res_virtual_gates.append(instr.operation)
        return res_virtual_gates
