from abc import ABC, abstractmethod
import itertools
from typing import Dict, Iterator, List, Tuple, Type

import networkx as nx
from qiskit.circuit.quantumcircuit import (
    QuantumCircuit,
    CircuitInstruction,
    Qubit,
    Clbit,
    Bit,
    CircuitInstruction,
)

from qvm import util

from .virtual_gate import VirtualBinaryGate, VirtualCZ, VirtualCX, VirtualRZZ


STANDARD_VIRTUAL_GATES = {"cz": VirtualCZ, "cx": VirtualCX, "rzz": VirtualRZZ}


class VirtualCircuitBase(ABC):
    @abstractmethod
    def circuit(self) -> QuantumCircuit:
        pass

    @abstractmethod
    def virtualize_gate(
        self, instr_index: int, virtual_gate_t: Type[VirtualBinaryGate]
    ) -> None:
        pass

    @abstractmethod
    def virtualize_connection(
        self,
        qubit1_index: int,
        qubit2_index: int,
        virtual_gates: Dict[str, Type[VirtualBinaryGate]],
    ) -> None:
        pass

    def connectivity_graph(self) -> nx.Graph:
        graph = nx.Graph()
        bb = nx.edge_betweenness_centrality(graph, normalized=False)
        nx.set_edge_attributes(graph, bb, "weight")
        graph.add_nodes_from(range(self.circuit().num_qubits))
        for instr in self.circuit().data:
            if isinstance(instr.operation, VirtualBinaryGate):
                continue
            if len(instr.qubits) >= 2:
                for qubit1, qubit2 in itertools.combinations(instr.qubits, 2):
                    qubit1_index = self.circuit().find_bit(qubit1).index
                    qubit2_index = self.circuit().find_bit(qubit2).index

                    if not graph.has_edge(qubit1_index, qubit2_index):
                        graph.add_edge(qubit1_index, qubit2_index, weight=0)
                    graph[qubit1_index][qubit2_index]["weight"] += 1
        return graph


class VirtualCircuit(VirtualCircuitBase):
    _circuit: QuantumCircuit

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit

    def circuit(self) -> QuantumCircuit:
        return self._circuit

    def virtualize_gate(
        self,
        instr_index: int,
        virtual_gate_t: Type[VirtualBinaryGate],
    ) -> None:
        old_instr = self._circuit.data[instr_index]
        virtual_gate = virtual_gate_t(old_instr.operation)
        new_instr = CircuitInstruction(virtual_gate, old_instr.qubits, old_instr.clbits)
        self._circuit.data[instr_index] = new_instr

    def virtualize_connection(
        self,
        qubit1_index: int,
        qubit2_index: int,
        virtual_gates: Dict[str, Type[VirtualBinaryGate]] = STANDARD_VIRTUAL_GATES,
    ) -> None:
        for i, instr in enumerate(self._circuit.data):
            qubit1 = self._circuit.qubits[qubit1_index]
            qubit2 = self._circuit.qubits[qubit2_index]
            if {qubit1, qubit2}.issubset(instr.qubits):
                self.virtualize_gate(
                    i, virtual_gate_t=virtual_gates[instr.operation.name]
                )

    def deflated(self) -> "VirtualCircuit":
        return VirtualCircuit(util.deflated_circuit(self._circuit))
