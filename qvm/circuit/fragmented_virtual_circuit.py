from dataclasses import dataclass
import itertools
from optparse import Option
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import networkx as nx

from qiskit.circuit.quantumcircuit import (
    Qubit,
    CircuitInstruction,
    QuantumRegister,
    Clbit,
    InstructionSet,
    Bit,
    ParameterValueType,
    QubitSpecifier,
    ClbitSpecifier,
    QuantumCircuit,
    Instruction,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit

from .virtual_gate.virtual_gate import VirtualBinaryGate
from .virtual_circuit import VirtualCircuit


class PartialVirtualGate(Instruction):
    _vgate: VirtualBinaryGate
    _index: int

    def __init__(self, virtual_gate: VirtualBinaryGate, qubit_index: int) -> None:
        self._vgate = virtual_gate
        self._index = qubit_index
        super().__init__(virtual_gate.name, 1, 1, [])

    @staticmethod
    def _circuit_on_qubit_index(
        circuit: QuantumCircuit, qubit_index: int
    ) -> QuantumCircuit:
        new_circuit = QuantumCircuit(1, 1)
        [
            new_circuit.append(
                CircuitInstruction(
                    instr.operation,
                    [new_circuit.qubits[0]],
                    [new_circuit.clbits[0] for _ in instr.clbit],
                )
            )
            for instr in circuit.data
            if circuit.find_bit(instr.qubits[0]).index == qubit_index
        ]
        return new_circuit

    def configure(self) -> List[QuantumCircuit]:
        return [
            self._circuit_on_qubit_index(circuit, self._index)
            for circuit in self._vgate.configure()
        ]


class Fragment:
    _original_circuit: VirtualCircuit
    _qubits: FrozenSet[Qubit]
    # the virtual gates that are virtualized as part of the original circuit
    _virtual_gates: Set[VirtualBinaryGate]

    def __init__(
        self,
        original_circuit: VirtualCircuit,
        qubits: Set[Qubit],
        frag_virtual_gates: Set[VirtualBinaryGate],
    ) -> None:
        self._original_circuit = original_circuit
        self._qubits = frozenset(qubits)
        self._virtual_gates = frag_virtual_gates

    def __hash__(self) -> int:
        return hash(self._qubits)

    @property
    def qubits(self) -> FrozenSet[Qubit]:
        return self._qubits

    @property
    def base_circuit(self) -> VirtualCircuit:
        res_circuit = VirtualCircuit(
            self._original_circuit.num_qubits, self._original_circuit.num_clbits
        )
        for instr in self._original_circuit.data:
            if set(instr.qubits).issubset(self._qubits):
                res_circuit.append(
                    CircuitInstruction(instr.operation, instr.qubits, instr.clbits)
                )
        return res_circuit.deflated()

    @property
    def virtual_gates(self) -> List[VirtualBinaryGate]:
        return [
            instr.operation
            for instr in self._original_circuit.data
            if isinstance(instr.operation, VirtualBinaryGate)
            and instr.operation in self._virtual_gates
        ]

    @property
    def config_ids(self) -> Iterator[Tuple[int, ...]]:
        conf_id_list: List[List[int]] = []
        for instr in self._original_circuit.data:
            if (
                isinstance(instr.operation, VirtualBinaryGate)
                and instr.operation in self._virtual_gates
            ):
                if instr.qubits[0] in self._qubits or instr.qubits[1] in self._qubits:
                    conf_id_list.append(list(range(len(instr.operation.configure()))))
                else:
                    conf_id_list.append([-1])
        return iter(itertools.product(*conf_id_list))

    def add_virtual_gate(self, virtual_gate: VirtualBinaryGate) -> None:
        if virtual_gate not in [
            instr.operation for instr in self._original_circuit.data
        ]:
            raise ValueError("Virtual gate not in circuit")
        self._virtual_gates.add(virtual_gate)

    def configured_fragments(self) -> Dict[Tuple[int, ...], VirtualCircuit]:
        conf_frags: Dict[Tuple[int, ...], VirtualCircuit] = {}
        for conf_id in self.config_ids:
            conf_frags[conf_id] = self._with_configuration(conf_id)
        return conf_frags

    def _with_configuration(self, config_id: Tuple[int, ...]) -> VirtualCircuit:
        if len(config_id) == 0:
            return self.base_circuit()

        new_circuit = VirtualCircuit(
            self._original_circuit.num_qubits,
            self._original_circuit.num_clbits + len(config_id),
        )

        conf_ctr = 0
        for instr in self._original_circuit.data:
            # if it is a virtual gate that has to be virtualized for the original circuit
            if (
                isinstance(instr.operation, VirtualBinaryGate)
                and instr.operation in self._virtual_gates
            ):
                if config_id[conf_ctr] == -1:
                    # if the virtual gate is not in the fragment at all
                    conf_ctr += 1
                    continue

                elif set(instr.qubits).issubset(self._qubits):
                    conf_instr = instr.operation.configure()[
                        config_id[conf_ctr]
                    ].to_instruction()
                    new_circuit.append(
                        CircuitInstruction(conf_instr, instr.qubits, instr.clbits)
                    )
                elif instr.qubits[0] in self._qubits:
                    conf_instr = (
                        PartialVirtualGate(instr.operation, 0)
                        .configure()[config_id[conf_ctr]]
                        .to_instruction()
                    )
                    new_circuit.append(
                        CircuitInstruction(conf_instr, instr.qubits, instr.clbits)
                    )
                elif instr.qubits[1] in self._qubits:
                    conf_instr = (
                        PartialVirtualGate(instr.operation, 1)
                        .configure()[config_id[conf_ctr]]
                        .to_instruction()
                    )
                    new_circuit.append(
                        CircuitInstruction(conf_instr, instr.qubits, instr.clbits)
                    )
                conf_ctr += 1

            else:
                new_circuit.append(
                    CircuitInstruction(instr.operation, instr.qubits, instr.clbits)
                )
        return new_circuit.deflated()


class FragmentedVirtualCircuit(VirtualCircuit):
    _fragments: Set[Fragment]

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        name: Optional[str] = None,
        global_phase: ParameterValueType = 0,
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(num_qubits, num_clbits, name, global_phase, metadata)
        self._fragments = {Fragment(self, set(self.qubits), set(self.virtual_gates))}

    @staticmethod
    def from_circuit(
        circuit: QuantumCircuit, fragment: bool = False
    ) -> "FragmentedVirtualCircuit":
        frag_circuit = FragmentedVirtualCircuit(
            circuit.num_qubits,
            circuit.num_clbits,
            circuit.name,
            circuit.global_phase,
            circuit.metadata,
        )
        for instr in circuit.data:
            frag_circuit.append(
                CircuitInstruction(instr.operation, instr.qubits, instr.clbits)
            )
        if fragment:
            frag_circuit.create_fragments()
        return frag_circuit

    @property
    def fragments(self) -> Set[Fragment]:
        return self._fragments

    @property
    def fragment_virtual_gates(self) -> Set[VirtualBinaryGate]:
        res_virtual_gates: Set[VirtualBinaryGate] = set()
        for instr in self.data:
            # if the virtual gate is not entirely in one fragment, it has to be
            # handeled by the original circuit
            if isinstance(instr.operation, VirtualBinaryGate) and not any(
                set(instr.qubits).issubset(frag.qubits) for frag in self._fragments
            ):
                res_virtual_gates.add(instr.operation)
        return res_virtual_gates

    def __hash__(self) -> int:
        return self.id

    def append(
        self,
        instruction: Union[Instruction, CircuitInstruction],
        qargs: Optional[Sequence[QubitSpecifier]] = None,
        cargs: Optional[Sequence[ClbitSpecifier]] = None,
    ) -> InstructionSet:
        if isinstance(instruction, VirtualBinaryGate):
            return super().append(instruction, qargs, cargs)
        qubits: List[Qubit]
        if isinstance(instruction, CircuitInstruction):
            qubits = list(instruction.qubits)
        elif isinstance(instruction, Instruction):
            qubits = list(
                itertools.chain(
                    *[self.qbit_argument_conversion(qarg) for qarg in qargs or []]
                )
            )
            print(qubits)

        affected_frags: List[Fragment] = [
            frag for frag in self._fragments if not set(qubits).isdisjoint(frag.qubits)
        ]
        if len(affected_frags) == 0:
            raise ValueError("No fragment affected by instruction")
        elif len(affected_frags) == 1:
            return super().append(instruction, qargs, cargs)
        else:
            merged = self.merge_fragments(affected_frags[0], affected_frags[1])
            for i in range(2, len(affected_frags)):
                merged = self.merge_fragments(merged, affected_frags[i])
        return super().append(instruction, qargs, cargs)

    def merge_fragments(self, fragment1: Fragment, fragment2: Fragment) -> Fragment:
        merged_qubits = fragment1.qubits | fragment2.qubits
        self._fragments.remove(fragment1)
        self._fragments.remove(fragment2)
        merged_frag = Fragment(
            self, set(merged_qubits), set(self.fragment_virtual_gates)
        )
        self._fragments.add(merged_frag)
        return merged_frag

    def create_fragments(self) -> None:
        self._fragments = set()
        qubit_sets = nx.connected_components(self.graph)
        for qubits in qubit_sets:
            self._fragments.add(
                Fragment(self, qubits, set(self.fragment_virtual_gates))
            )
