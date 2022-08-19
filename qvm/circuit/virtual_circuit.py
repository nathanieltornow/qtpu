import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import networkx as nx
from qiskit.circuit.quantumcircuit import (
    QuantumCircuit,
    Instruction,
    CircuitInstruction,
    Qubit,
    Bit,
    CircuitInstruction,
    ParameterValueType,
    QubitSpecifier,
    ClbitSpecifier,
    InstructionSet,
)

from qiskit.converters import circuit_to_dag

from .virtual_gate import VirtualBinaryGate, VirtualCZ, VirtualCX, VirtualRZZ


STANDARD_VIRTUAL_GATES = {"cz": VirtualCZ, "cx": VirtualCX, "rzz": VirtualRZZ}


class VirtualCircuit(QuantumCircuit):

    _connectivity_graph: nx.Graph

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        name: Optional[str] = None,
        global_phase: ParameterValueType = 0,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(
            num_qubits,
            num_clbits,
            name=name,
            global_phase=global_phase,
            metadata=metadata,
        )
        # initialize connectivity graph
        self._connectivity_graph = nx.Graph()
        bb = nx.edge_betweenness_centrality(self._connectivity_graph, normalized=False)
        nx.set_edge_attributes(self._connectivity_graph, bb, "weight")
        self._connectivity_graph.add_nodes_from(self.qubits)

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and isinstance(other, VirtualCircuit)

    @staticmethod
    def from_circuit(circuit: QuantumCircuit) -> "VirtualCircuit":
        res = VirtualCircuit(
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
            name=circuit.name,
            global_phase=circuit.global_phase,
            metadata=circuit.metadata,
        )
        for instr in circuit.data:
            instr = CircuitInstruction(
                instr.operation,
                [res.qubits[circuit.find_bit(qubit)[0]] for qubit in instr.qubits],
                [res.clbits[circuit.find_bit(clbit)[0]] for clbit in instr.clbits],
            )
            res.append(instr)
        return res

    def copy(self, name: Optional[str] = None) -> QuantumCircuit:
        cp = super().copy(name)
        return VirtualCircuit.from_circuit(cp)

    @property
    def graph(self) -> nx.Graph:
        return self._connectivity_graph.copy()

    @property
    def virtual_gates(self) -> List[VirtualBinaryGate]:
        return [
            cinstr.operation
            for cinstr in self.data
            if isinstance(cinstr.operation, VirtualBinaryGate)
        ]

    @property
    def config_ids(self) -> List[Tuple[int, ...]]:
        """
        Gets all configuration ids that for the circuits that need to be
        constructed.

        Returns:
            List[Tuple[int, ...]]: The configuration ids
        """
        virt_gates = self.virtual_gates
        if len(virt_gates) == 0:
            return []
        result: List[Tuple[int, ...]] = [
            (i,) for i in range(len(virt_gates[0].configure()))
        ]
        for vgate in virt_gates[1:]:
            confs = [(i,) for i in range(len(vgate.configure()))]
            for _ in range(len(result)):
                old_conf = result.pop(0)
                for conf in confs:
                    result.append(old_conf + conf)
        return result

    def deflated(self) -> "VirtualCircuit":
        # determine active qubits
        dag = circuit_to_dag(self)
        sorted_active_qubits: List[Qubit] = sorted(
            [qubit for qubit in self.qubits if qubit not in dag.idle_wires()],
            key=lambda q: self.find_bit(q).index,
        )

        # map the bits to the bits of the new circuits
        new_vc = VirtualCircuit(len(sorted_active_qubits), self.num_clbits)
        bit_map: Dict[Bit, Bit] = {
            q: new_vc.qubits[i] for i, q in enumerate(sorted_active_qubits)
        }
        bit_map |= {c: new_vc.clbits[i] for i, c in enumerate(self.clbits)}

        # copy the instrcution to the new circuit
        for instr in self.data:
            new_vc.append(
                CircuitInstruction(
                    instr.operation,
                    [bit_map[q] for q in instr.qubits],
                    [bit_map[c] for c in instr.clbits],
                )
            )
        return new_vc

    def append(
        self,
        instruction: Union[Instruction, CircuitInstruction],
        qargs: Optional[Sequence[QubitSpecifier]] = None,
        cargs: Optional[Sequence[ClbitSpecifier]] = None,
    ) -> InstructionSet:
        super().append(instruction, qargs, cargs)
        self._add_edges_from_instr(self.data[-1])

    def virtualize_gate(
        self, gate_index: int, virtual_gate_t: Type[VirtualBinaryGate]
    ) -> None:
        old_instr = self._data[gate_index]
        virtual_gate = virtual_gate_t(old_instr.operation)
        new_instr = CircuitInstruction(virtual_gate, old_instr.qubits, old_instr.clbits)
        self._data[gate_index] = new_instr
        self._decrement_egde_weight(old_instr.qubits[0], old_instr.qubits[1])

    def virtualize_connection(
        self,
        qubit1: Qubit,
        qubit2: Qubit,
        virtual_gates: Dict[str, Type[VirtualBinaryGate]] = STANDARD_VIRTUAL_GATES,
    ) -> None:
        for i, instr in enumerate(self.data):
            if {qubit1, qubit2}.issubset(instr.qubits):
                self.virtualize_gate(i, virtual_gates[instr.operation.name])

    def configured_circuits(
        self,
    ) -> Dict[Tuple[int, ...], QuantumCircuit]:
        conf_circs: Dict[Tuple[int, ...], QuantumCircuit] = {}
        for conf_id in self.config_ids:
            conf_circs[conf_id] = self._with_inserted_config(conf_id)
        if len(conf_circs) == 0:
            return {(): self.copy()}
        return conf_circs

    def _with_inserted_config(self, conf_id: Tuple[int, ...]) -> QuantumCircuit:
        num_clbits = self.num_clbits + sum(
            vgate.num_clbits for vgate in self.virtual_gates
        )
        circuit = QuantumCircuit(self.num_qubits, num_clbits)
        conf_index = 0
        for i in range(len(self.data)):
            cinstr = self.data[i]
            if isinstance(cinstr.operation, VirtualBinaryGate):
                instruction = cinstr.operation.configure()[
                    conf_id[conf_index]
                ].to_instruction(label=f"config{conf_id[conf_index]}")
                clbit = circuit.clbits[self.num_clbits + conf_index]
                print(instruction, cinstr.qubits, [clbit])
                circuit.append(CircuitInstruction(instruction, cinstr.qubits, [clbit]))
                conf_index += 1
            else:
                circuit.append(
                    CircuitInstruction(cinstr.operation, cinstr.qubits, cinstr.clbits)
                )
        return circuit

    def _add_edges_from_instr(self, circuit_instruction: CircuitInstruction) -> None:
        if isinstance(circuit_instruction.operation, VirtualBinaryGate):
            return
        qubits = circuit_instruction.qubits
        if len(qubits) < 2:
            return
        for qubit1, qubit2 in itertools.combinations(qubits, 2):
            self._add_edge(qubit1, qubit2)
            print("adding edge", qubit1, qubit2)

    def _add_edge(self, u: Qubit, v: Qubit) -> None:
        if self._connectivity_graph.has_edge(u, v):
            self._connectivity_graph[u][v]["weight"] += 1
        else:
            self._connectivity_graph.add_edge(u, v, weight=1)

    def _decrement_egde_weight(self, u: Qubit, v: Qubit) -> None:
        print("decrementing edge", u, v)
        if self._connectivity_graph.has_edge(u, v):
            if self._connectivity_graph[u][v]["weight"] > 1:
                self._connectivity_graph[u][v]["weight"] -= 1
            else:
                self._connectivity_graph.remove_edge(u, v)

    ###########################################################################
    # Deactivate the following methods as virtual circuit cannot resize for now
    ###########################################################################

    # def add_bits(self, bits: Iterable[Bit]) -> None:
    #     raise NotImplementedError("virtual circuit is immutable")

    # def add_register(self, *regs: Union[Register, int, Sequence[Bit]]) -> None:
    #     raise NotImplementedError("virtual circuit is immutable")
