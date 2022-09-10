from typing import Dict, List, Optional, Sequence, Set, Union

from qiskit.circuit.quantumcircuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    Qubit,
)
from qiskit.providers import Backend
import networkx as nx

from qvm.converters import circuit_to_connectivity_graph
from qvm.virtual_gate.virtual_gate import VirtualBinaryGate


class MappedRegister(QuantumRegister):
    backend: Backend
    initial_layout: Optional[List[int]]

    def __init__(
        self,
        qreg: QuantumRegister,
        backend: Backend,
        initial_layout: Optional[List[int]],
    ):
        self.backend = backend
        self.initial_layout = initial_layout
        super().__init__(qreg.size, qreg.name, None)


class DistributedCircuit(QuantumCircuit):
    @staticmethod
    def from_circuit(
        circuit: QuantumCircuit, qubit_groups: Optional[List[Set[Qubit]]] = None
    ) -> "DistributedCircuit":
        if qubit_groups is not None:
            # check qubit-groups
            if set().union(*qubit_groups) != set(circuit.qubits) or bool(
                set().intersection(*qubit_groups)
            ):
                raise ValueError("qubit-groups not valid")

        else:
            con_graph = circuit_to_connectivity_graph(circuit)
            qubit_groups = list(nx.connected_components(con_graph))

        new_frags = [
            QuantumRegister(len(nodes), name=f"frag_{i}")
            for i, nodes in enumerate(qubit_groups)
        ]
        qubit_map: Dict[Qubit, Qubit] = {}  # old -> new Qubit
        for nodes, circ in zip(qubit_groups, new_frags):
            node_l = list(nodes)
            for i in range(len(node_l)):
                qubit_map[node_l[i]] = circ[i]

        vc = DistributedCircuit(
            *new_frags,
            *circuit.cregs,
            name=circuit.name,
            global_phase=circuit.global_phase,
            metadata=circuit.metadata,
        )

        for circ_instr in circuit.data:
            vc.append(
                circ_instr.operation,
                [qubit_map[q] for q in circ_instr.qubits],
                circ_instr.clbits,
            )
        return vc

    @property
    def fragments(self) -> List[QuantumRegister]:
        return self.qregs

    @property
    def num_fragments(self) -> int:
        return len(self.qregs)

    @property
    def virtual_gates(self) -> List[VirtualBinaryGate]:
        return [
            instr.operation
            for instr in self.data
            if isinstance(instr.operation, VirtualBinaryGate)
        ]

    def is_valid(self) -> bool:
        for circ_instr in self.data:
            if (
                len(circ_instr.qubits) > 2
                and not isinstance(circ_instr.operation, VirtualBinaryGate)
                and len(set(qubit.register for qubit in circ_instr.qubits)) > 1
            ):
                return False
        return True

    def add_config_register(self, size: int) -> ClassicalRegister:
        num_conf_register = sum(
            1 for creg in self.cregs if creg.name.startswith("conf")
        )
        reg = ClassicalRegister(size, name=f"conf_{num_conf_register}")
        self.add_register(reg)
        return reg

    def map_fragment(
        self,
        fragment: QuantumRegister,
        backend: Backend,
        initial_layout: Optional[List[int]] = None,
    ) -> None:
        mapped_register = MappedRegister(fragment, backend, initial_layout)
        reg_index = self.qregs.index(fragment)
        self.qregs[reg_index] = mapped_register

    def fragment_as_circuit(self, fragment: QuantumRegister) -> QuantumCircuit:
        circ = QuantumCircuit(fragment, *self.cregs)
        for instr in self.data:
            if set(instr.qubits) <= set(fragment):
                circ.append(instr.operation, instr.qubits, instr.clbits)
        return circ
