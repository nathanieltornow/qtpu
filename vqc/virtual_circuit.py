import itertools
from typing import Iterator, Dict, Tuple

import networkx as nx
from qiskit.circuit import (
    QuantumCircuit,
    Barrier,
    QuantumRegister,
    Qubit,
    Instruction,
    ClassicalRegister,
)

from vqc.util import _circuit_to_connectivity_graph, _circuit_on_index
from vqc.virtual_gate import VirtualGate


class Placeholder(Instruction):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, num_qubits=1, num_clbits=1)


InputType = Dict[str, QuantumCircuit]
ConfigIdType = Tuple[int, ...]


class AbstractCircuit(QuantumCircuit):
    def insert_placeholders(self, input: InputType) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.qregs, *self.cregs, name=self.name)
        for instr in self.data:
            op, qubits, clbits = (instr.operation, instr.qubits, instr.clbits)
            if isinstance(op, Placeholder):
                if op.name not in input:
                    raise ValueError(f"Missing input {op.name}")
                circuit.append(input[op.name].to_instruction(), qubits, clbits)
            else:
                circuit.append(op, qubits, clbits)
        return circuit


class VirtualCircuit(QuantumCircuit):
    def __init__(self, circuit: QuantumCircuit) -> None:
        con_graph = _circuit_to_connectivity_graph(circuit)
        qubit_groups = list(nx.connected_components(con_graph))

        new_frags = [
            QuantumRegister(len(nodes), name=f"frag{i}")
            for i, nodes in enumerate(qubit_groups)
        ]
        qubit_map: dict[Qubit, Qubit] = {}  # old -> new Qubit
        for nodes, circ in zip(qubit_groups, new_frags):
            node_l = list(nodes)
            for i in range(len(node_l)):
                qubit_map[node_l[i]] = circ[i]

        super().__init__(
            *new_frags,
            *circuit.cregs,
            name=circuit.name,
            global_phase=circuit.global_phase,
            metadata=circuit.metadata,
        )
        vgate_cnt = 0
        for instr in circuit.data:
            if isinstance(instr.operation, VirtualGate):
                vgate_cnt += 1
            self.append(
                instr.operation,
                [qubit_map[q] for q in instr.qubits],
                instr.clbits,
            )
        config_register = ClassicalRegister(vgate_cnt, name="config")
        self.add_register(config_register)

    def abstract_circuit(self, fragment: QuantumRegister) -> AbstractCircuit:
        abstr_circ = AbstractCircuit(fragment, *self.cregs, name=self.name)
        conf_ctr = 0
        for instr in self.data:
            op, qubits, clbits = (instr.operation, instr.qubits, instr.clbits)
            if isinstance(op, VirtualGate):
                if qubits[0] in fragment:
                    abstr_circ.append(
                        Placeholder(f"config{conf_ctr}_0"), (qubits[0],), clbits
                    )
                if qubits[1] in fragment:
                    abstr_circ.append(
                        Placeholder(f"config{conf_ctr}_1"), (qubits[1],), clbits
                    )
                conf_ctr += 1
            elif set(qubits) <= set(fragment):
                abstr_circ.append(op, qubits, clbits)
        return abstr_circ

    def _config_ids(self, fragment: QuantumRegister) -> Iterator[ConfigIdType]:
        vgate_instrs = [
            instr for instr in self.data if isinstance(instr.operation, VirtualGate)
        ]
        conf_l = [
            tuple(range(len(instr.operation.configure())))
            if set(instr.qubits) & set(fragment)
            else (-1,)
            for instr in vgate_instrs
        ]
        return iter(itertools.product(*conf_l))

    def fragment_inputs(
        self, fragment: QuantumRegister
    ) -> Iterator[Tuple[ConfigIdType, InputType]]:
        vgate_instrs = [
            instr for instr in self.data if isinstance(instr.operation, VirtualGate)
        ]
        for conf_id in self._config_ids(fragment):
            input_: InputType = {}
            for i, conf in enumerate(conf_id):
                if conf == -1:
                    continue
                op, qubits = (
                    vgate_instrs[i].operation,
                    vgate_instrs[i].qubits,
                )
                if qubits[0] in fragment:
                    input_[f"config{i}_0"] = _circuit_on_index(
                        op.configurations(conf), 0
                    )
                if qubits[1] in fragment:
                    input_[f"config{i}_1"] = _circuit_on_index(
                        op.configurations(conf), 1
                    )
            yield conf_id, input_
