import itertools
from typing import Dict, List, Iterator, Tuple

import networkx as nx
from qiskit.circuit import (
    QuantumCircuit,
    ClassicalRegister,
    QuantumRegister,
    Qubit,
    Instruction,
    Barrier,
)

from vqc.util import circuit_to_connectivity_graph, circuit_on_index
from vqc.types import VirtualGate, InputType, ConfigIdType


class Placeholder(Instruction):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, num_qubits=1, num_clbits=1, params=[])


class AbstractCircuit(QuantumCircuit):
    """AbstractCircuit is a QuantumCircuit with placeholders."""

    def insert_placeholders(self, input: InputType) -> QuantumCircuit:
        """Inserts placeholders into the circuit for the given input.

        Args:
            input (InputType): The input (gate_name -> input_circuit)

        Raises:
            ValueError: If a placeholder does not have a corresponding input.

        Returns:
            QuantumCircuit: The circuit with placeholders inserted
                (no placeholders in the circuit anymore).
        """
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


class Fragment(QuantumRegister):
    """Fragment of a virtual circuit is moddeled as a QuantumRegister."""

    pass


class VirtualCircuit(QuantumCircuit):
    def __init__(self, circuit: QuantumCircuit):
        con_graph = circuit_to_connectivity_graph(circuit)
        qubit_groups = list(nx.connected_components(con_graph))

        new_frags = [
            Fragment(len(nodes), name=f"frag{i}")
            for i, nodes in enumerate(qubit_groups)
        ]
        qubit_map: Dict[Qubit, Qubit] = {}  # old -> new Qubit
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

        for circ_instr in circuit.data:
            self.append(
                circ_instr.operation,
                [qubit_map[q] for q in circ_instr.qubits],
                circ_instr.clbits,
            )

        conf_reg = ClassicalRegister(len(self.virtual_gates), name="conf")
        self.add_register(conf_reg)
        self._conf_reg = conf_reg

    @property
    def fragments(self) -> List[Fragment]:
        return self.qregs

    @property
    def virtual_gates(self) -> List[VirtualGate]:
        return [
            instr.operation
            for instr in self.data
            if isinstance(instr.operation, VirtualGate)
        ]

    @property
    def is_valid(self) -> bool:
        """Checks if the virtual circuit is valid and fragments are not connected."""

        for circ_instr in self.data:
            if (
                len(circ_instr.qubits) >= 2
                and not isinstance(circ_instr.operation, Barrier)
                and len(
                    [
                        True
                        for frag in self.fragments
                        if set(circ_instr.qubits) & set(frag)
                    ]
                )
                > 1
            ):
                return False
        return True

    def abstract_circuit(self, fragment: Fragment) -> AbstractCircuit:
        """Returns the abstract circuit for the given fragment.

        Args:
            fragment (Fragment): The fragment for which the abstract circuit is returned.

        Returns:
            AbstractCircuit: The abstract circuit for the given fragment.
        """

        abstr_circ = AbstractCircuit(fragment, *self.cregs, name=self.name)
        conf_ctr = 0
        for instr in self.data:
            op, qubits, clbits = (instr.operation, instr.qubits, instr.clbits)
            if isinstance(op, VirtualGate):
                if qubits[0] in fragment:
                    abstr_circ.append(
                        Placeholder(f"config_{conf_ctr}_0"),
                        (qubits[0],),
                        (self._conf_reg[conf_ctr],),
                    )
                if qubits[1] in fragment:
                    abstr_circ.append(
                        Placeholder(f"config_{conf_ctr}_1"),
                        (qubits[1],),
                        (self._conf_reg[conf_ctr],),
                    )
                conf_ctr += 1
            elif set(qubits) <= set(fragment):
                abstr_circ.append(op, qubits, clbits)
        return abstr_circ

    def inputs(self, fragment: Fragment) -> Iterator[Tuple[ConfigIdType, InputType]]:
        """Generates all inputs for the given fragment that need to be evaluated.

        Args:
            fragment (Fragment): The fragment for which the inputs are generated.

        Yields:
            Iterator[Tuple[ConfigIdType, InputType]]: The inputs for the given fragment.
        """

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
                    input_[f"config_{i}_0"] = circuit_on_index(
                        op.configurations(conf), 0
                    )
                if qubits[1] in fragment:
                    input_[f"config_{i}_1"] = circuit_on_index(
                        op.configurations(conf), 1
                    )
            yield conf_id, input_

    def _config_ids(self, fragment: Fragment) -> Iterator[ConfigIdType]:
        """Generates all possible configuration ids for the given fragment.

        Args:
            fragment (Fragment): The fragment for which the configuration ids are generated.

        Yields:
            Iterator[ConfigIdType]: All possible configuration ids for the given fragment.
        """

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
