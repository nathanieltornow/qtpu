from typing import Iterator
import itertools

import networkx as nx
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    CircuitInstruction,
    Qubit,
)


class DAG(nx.MultiDiGraph):
    def __init__(self, circuit: QuantumCircuit):
        def _next_op_on_qubit(qubit: int, from_idx: int) -> int:
            for i, instr in enumerate(circuit[from_idx + 1 :]):
                if qubit in instr.qubits:
                    return i + from_idx + 1
            return -1

        super().__init__()

        for i, instr in enumerate(circuit):
            self.add_node(i, instr=instr)
            for qubit in instr.qubits:
                next_op = _next_op_on_qubit(qubit, i)
                if next_op > -1:
                    self.add_edge(i, next_op)

        self._qregs = circuit.qregs
        self._cregs = circuit.cregs

    @property
    def qubits(self) -> list[Qubit]:
        return list(itertools.chain(*self._qregs))

    @property
    def clbits(self) -> list[Qubit]:
        return list(itertools.chain(*self._cregs))

    @property
    def depth(self) -> int:
        return nx.dag_longest_path_length(self)

    def to_circuit(self) -> QuantumCircuit:
        order = list(nx.topological_sort(self))
        circuit = QuantumCircuit(*self._qregs, *self._cregs)
        for i in order:
            instr = self.nodes[i]["instr"]
            circuit.append(instr)
        return circuit

    def add_instr_node(self, instr: CircuitInstruction) -> int:
        new_id = self.number_of_nodes()
        self.add_node(new_id, instr=instr)
        return new_id

    def get_node_instr(self, node: int) -> CircuitInstruction:
        return self.nodes[node]["instr"]

    def compact(self) -> None:
        # find the qubits not used
        used_qubits: set[Qubit] = set()
        for node in self.nodes:
            used_qubits.update(self.get_node_instr(node).qubits)
        unused_qubits = set(itertools.chain(*self._qregs)) - used_qubits
        if len(unused_qubits) == 0:
            return

        new_qreg = QuantumRegister(len(used_qubits), "q")
        qubit_mapping: dict[Qubit, Qubit] = {
            qubit: new_qreg[i] for i, qubit in enumerate(used_qubits)
        }
        # update the circuit
        for node in self.nodes:
            instr = self.get_node_instr(node)
            new_qubits = [qubit_mapping[qubit] for qubit in instr.qubits]
            instr.qubits = new_qubits

        self._qregs = [new_qreg]

    def instructions_on_qubit(self, qubit: Qubit) -> Iterator[CircuitInstruction]:
        for node in nx.topological_sort(self):
            instr = self.get_node_instr(node)
            if qubit in instr.qubits:
                yield instr

    def nodes_on_qubit(self, qubit: Qubit) -> Iterator[int]:
        for node in nx.topological_sort(self):
            instr = self.get_node_instr(node)
            if qubit in instr.qubits:
                yield node
