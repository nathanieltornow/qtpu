from abc import ABC, abstractmethod
from importlib import metadata
import itertools
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, Union

import networkx as nx
from qiskit.circuit.quantumcircuit import (
    QuantumCircuit,
    CircuitInstruction,
    Qubit,
    CircuitInstruction,
    ParameterValueType,
    Register,
    Bit,
    QuantumRegister,
)
from qiskit.converters import circuit_to_dag


from .virtual_gate import VirtualBinaryGate, VirtualCZ, VirtualCX, VirtualRZZ


STANDARD_VIRTUAL_GATES = {"cz": VirtualCZ, "cx": VirtualCX, "rzz": VirtualRZZ}


class VirtualCircuitInterface(ABC):
    @abstractmethod
    def virtualize_connection(
        self,
        qubit1: Qubit,
        qubit2: Qubit,
        virtual_gates: Dict[str, Type[VirtualBinaryGate]],
    ) -> None:
        pass

    @abstractmethod
    def connectivity_graph(self) -> nx.Graph:
        pass


class Fragment(VirtualCircuitInterface):
    _vc: VirtualCircuitInterface
    _qubits: Set[Qubit]
    _ids = itertools.count(0)

    def __init__(
        self, virtual_circuit: VirtualCircuitInterface, qubits: Set[Qubit]
    ) -> None:
        self._vc = virtual_circuit
        self._qubits = qubits
        self.id = next(self._ids)

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other):
        return isinstance(other, Fragment) and self._qubits == other._qubits

    def __repr__(self) -> str:
        return f"Fragment({self._qubits})"

    @property
    def qubits(self) -> Set[Qubit]:
        return self._qubits

    def virtualize_connection(
        self,
        qubit1: Qubit,
        qubit2: Qubit,
        virtual_gates: Dict[str, Type[VirtualBinaryGate]] = STANDARD_VIRTUAL_GATES,
    ) -> None:
        return self._vc.virtualize_connection(qubit1, qubit2, virtual_gates)

    def connectivity_graph(self) -> nx.Graph:
        return nx.subgraph(self._vc.connectivity_graph(), self._qubits)


class VirtualCircuit(QuantumCircuit, VirtualCircuitInterface):
    @staticmethod
    def from_circuit(circuit: QuantumCircuit) -> "VirtualCircuit":
        vc = VirtualCircuit(
            *circuit.qregs.copy(),
            *circuit.cregs.copy(),
            name=circuit.name,
            global_phase=circuit.global_phase,
            metadata=circuit.metadata,
        )
        for circ_instr in circuit.data:
            vc.append(circ_instr.copy())
        return vc

    @property
    def fragments(self) -> Set[Fragment]:
        return {
            Fragment(self, qubits)
            for qubits in nx.connected_components(self.connectivity_graph())
        }

    def to_circuit(self) -> "QuantumCircuit":
        return self.decompose()

    def copy(self, name: Optional[str] = None) -> "VirtualCircuit":
        cp = super().copy(name)
        return self.from_circuit(cp)

    def virtualize_connection(
        self,
        qubit1: Qubit,
        qubit2: Qubit,
        virtual_gates: Dict[str, Type[VirtualBinaryGate]] = STANDARD_VIRTUAL_GATES,
        fragmenting: bool = True,
    ) -> None:
        assert qubit1 != qubit2
        for i in range(len(self.data)):
            circ_instr = self.data[i]
            if {qubit1, qubit2}.issubset(circ_instr.qubits):
                virtual_gate = virtual_gates[circ_instr.operation.name](
                    circ_instr.operation, fragmenting
                )
                self.data[i] = CircuitInstruction(virtual_gate, circ_instr.qubits, [])

    def connectivity_graph(self) -> nx.Graph:
        graph = nx.Graph()
        bb = nx.edge_betweenness_centrality(graph, normalized=False)
        nx.set_edge_attributes(graph, bb, "weight")
        graph.add_nodes_from(self.qubits)
        for instr in self.data:
            if (
                isinstance(instr.operation, VirtualBinaryGate)
                and instr.operation.is_fragmenting
            ):
                continue
            if len(instr.qubits) >= 2:
                for qubit1, qubit2 in itertools.combinations(instr.qubits, 2):
                    if not graph.has_edge(qubit1, qubit2):
                        graph.add_edge(qubit1, qubit2, weight=0)
                    graph[qubit1][qubit2]["weight"] += 1
        return graph

    def deflated(self, qubits: Optional[Set[Qubit]] = None) -> "VirtualCircuit":
        if qubits is None:
            dag = circuit_to_dag(self)
            qubits = set(
                qubit for qubit in self.qubits if qubit not in dag.idle_wires()
            )

        qreg = QuantumRegister(bits=qubits)
        new_circuit = VirtualCircuit(qreg, *self.cregs)
        sorted_qubits = sorted(qubits, key=lambda q: self.find_bit(q).index)
        qubit_map: Dict[Qubit, Qubit] = {
            q: new_circuit.qubits[i] for i, q in enumerate(sorted_qubits)
        }

        for circ_instr in self.data:
            if set(circ_instr.qubits) <= qubits:
                new_circuit.append(
                    circ_instr.operation,
                    [qubit_map[q] for q in circ_instr.qubits],
                    circ_instr.clbits,
                )
        return new_circuit
