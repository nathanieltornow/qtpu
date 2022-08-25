from abc import ABC, abstractmethod
import itertools
import secrets
import string
from typing import Iterator, List, Optional, Set, Tuple
import uuid

from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister, Qubit

from qvm.circuit.virtual_circuit import VirtualBinaryGate
from qvm.circuit import VirtualCircuit


def unique_name() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(secrets.choice(alphabet) for i in range(10))


class ConfiguratorInterface(ABC):
    @abstractmethod
    def virtual_gates(self) -> List[VirtualBinaryGate]:
        pass


class VirtualCircuitConfigurator(ConfiguratorInterface):
    _vc: VirtualCircuit

    def __init__(self, virtual_circuit: VirtualCircuit):
        self._vc = virtual_circuit

    def __iter__(self) -> Iterator[QuantumCircuit]:
        return self._configured_circuits()

    def virtual_gates(self) -> List[VirtualBinaryGate]:
        return [
            instr.operation
            for instr in self._vc.data
            if isinstance(instr.operation, VirtualBinaryGate)
        ]

    def _config_ids(self) -> Iterator[Tuple[int, ...]]:
        conf_list: List[Tuple[int, ...]] = [
            tuple(range(len(vgate.configure()))) for vgate in self.virtual_gates()
        ]
        return iter(itertools.product(*conf_list))

    def _configured_circuits(self) -> Iterator[QuantumCircuit]:
        for config_id in self._config_ids():
            yield self._circuit_with_config(config_id)

    def _circuit_with_config(self, conf_id: Tuple[int, ...]) -> QuantumCircuit:
        if len(conf_id) != len(self.virtual_gates()):
            raise ValueError("config length does not match virtual gate length")
        if len(conf_id) == 0:
            return self._vc.to_circuit()

        conf_circuit = self._vc.copy()
        conf_reg = ClassicalRegister(size=len(conf_id), name=unique_name())
        conf_circuit.add_register(conf_reg)

        conf_ctr = 0
        for i in range(len(conf_circuit)):
            circ_instr = conf_circuit.data[i]
            operation = circ_instr.operation
            if isinstance(operation, VirtualBinaryGate):
                conf_op = operation.configure()[conf_id[conf_ctr]].to_instruction()
                new_instr = CircuitInstruction(
                    conf_op, circ_instr.qubits, [conf_reg[conf_ctr]]
                )
                conf_circuit.data[i] = new_instr
                conf_ctr += 1
        return conf_circuit.to_circuit()


class BinaryFragmentsConfigurator(ConfiguratorInterface):
    _vc: VirtualCircuit
    _A: Set[Qubit]
    _B: Set[Qubit]

    def __init__(
        self,
        vc: VirtualCircuit,
        A: Optional[Set[Qubit]] = None,
        B: Optional[Set[Qubit]] = None,
    ) -> None:
        self._vc = vc
        if A is not None and B is not None:
            # TODO error checking
            self._A = A
            self._B = B
        else:
            if len(vc.fragments) != 2:
                raise ValueError("virtual circuit must have two fragments")
            frag = vc.fragments
            self._A = frag.pop().qubits
            self._B = frag.pop().qubits

    def __iter__(
        self,
    ) -> Iterator[Tuple[VirtualCircuit, VirtualCircuit]]:
        return self.configured_fragments()

    def virtual_gates(self) -> List[VirtualBinaryGate]:
        return [instr.operation for instr in self.virtual_instructions()]

    def virtual_instructions(self) -> List[CircuitInstruction]:
        return [
            instr
            for instr in self._vc.data
            if isinstance(instr.operation, VirtualBinaryGate)
            and (
                (instr.qubits[0] in self._A and instr.qubits[1] in self._B)
                or (instr.qubits[1] in self._A and instr.qubits[0] in self._B)
            )
        ]

    def _config_ids(self) -> Iterator[Tuple[int, ...]]:
        conf_list: List[Tuple[int, ...]] = [
            tuple(range(len(vinstr.operation.configure())))
            for vinstr in self.virtual_instructions()
        ]
        return iter(itertools.product(*conf_list))

    @staticmethod
    def _append_if_in(
        circuit: QuantumCircuit, circ_instr: CircuitInstruction, qubits: Set[Qubit]
    ) -> bool:
        if set(circ_instr.qubits) <= qubits:
            circuit.append(circ_instr)
            return True
        return False

    def _frags_with_config(
        self, config_id: Tuple[int, ...]
    ) -> Tuple[VirtualCircuit, VirtualCircuit]:

        conf_reg = ClassicalRegister(len(config_id), name=unique_name())
        conf_circuit = VirtualCircuit(
            *self._vc.qregs.copy(), *self._vc.cregs.copy(), conf_reg
        )

        conf_ctr = 0
        for instr in self._vc.data:
            if self._append_if_in(conf_circuit, instr, self._A):
                continue
            if self._append_if_in(conf_circuit, instr, self._B):
                continue

            if not isinstance(instr.operation, VirtualBinaryGate):
                raise Exception("something went wrong")

            config = instr.operation.configure()[config_id[conf_ctr]].to_instruction()
            conf_circuit.append(config, instr.qubits, [conf_reg[conf_ctr]])

        conf_circuit = VirtualCircuit.from_circuit(conf_circuit.decompose())
        return conf_circuit.deflated(self._A), conf_circuit.deflated(self._B)

    def configured_fragments(
        self,
    ) -> Iterator[Tuple[VirtualCircuit, VirtualCircuit]]:
        for config_id in self._config_ids():
            yield self._frags_with_config(config_id)

    # # ------ Methods to create all configurations for executing the virtual circuit

    # def _config_ids(self) -> Iterator[Tuple[int, ...]]:
    #     conf_list: List[Tuple[int, ...]] = [
    #         tuple(range(len(vinstr.operation.configure())))
    #         for vinstr in self.virtual_instructions()
    #     ]
    #     return iter(itertools.product(*conf_list))

    # def _circuit_with_config(self, config_id: Tuple[int, ...]) -> QuantumCircuit:
    #     vgates = [instr.operation for instr in self.virtual_instructions()]
    #     if len(config_id) != len(vgates):
    #         raise ValueError("config length does not match virtual gate length")
    #     if len(config_id) == 0:
    #         return self.to_circuit()

    #     conf_circuit = self.copy()
    #     conf_reg = ClassicalRegister(size=len(config_id), name="config")
    #     conf_circuit.add_register(conf_reg)

    #     conf_ctr = 0
    #     for i in range(len(conf_circuit)):
    #         circ_instr = conf_circuit.data[i]
    #         operation = circ_instr.operation
    #         if isinstance(operation, VirtualBinaryGate) and operation in vgates:
    #             conf_op = operation.configure()[config_id[conf_ctr]].to_instruction()
    #             new_instr = CircuitInstruction(
    #                 conf_op, circ_instr.qubits, conf_reg[conf_ctr]
    #             )
    #             conf_circuit.data[i] = new_instr
    #             conf_ctr += 1
    #     return conf_circuit.to_circuit()

    # def configured_circuits(self) -> Iterator[Tuple[Tuple[int, ...], QuantumCircuit]]:
    #     for config_id in self._config_ids():
    #         yield config_id, self._circuit_with_config(config_id)
