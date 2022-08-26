import itertools
import secrets
from typing import Iterator, List, Set, Tuple

from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister, Qubit

from qvm.circuit.virtual_circuit import Fragment, VirtualBinaryGate
from qvm.circuit import VirtualCircuit


def unique_name() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(secrets.choice(alphabet) for i in range(10))


class VirtualCircuitConfigurator:
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


class FragmentConfigurator:
    def __init__(self, vc: VirtualCircuit, fragment: Fragment) -> None:
        if fragment not in vc.fragments:
            raise ValueError("fragment not in virtual circuit")
        self._vc = vc
        self._fragment = fragment

    def virtual_instr(self) -> List[CircuitInstruction]:
        return [
            instr
            for instr in self._vc.data
            if isinstance(instr.operation, VirtualBinaryGate)
            and not any(set(instr.qubits) <= frag.qubits for frag in self._vc.fragments)
        ]

    def virtual_gates(self) -> List[VirtualBinaryGate]:
        return [instr.operation for instr in self.virtual_instr()]

    def all_config_ids(self) -> Iterator[Tuple[int, ...]]:
        conf_list = [
            tuple(range(len(vgate.configure()))) for vgate in self.virtual_gates()
        ]
        return iter(itertools.product(*conf_list))

    def _config_ids(self) -> Iterator[Tuple[int, ...]]:
        conf_list: List[Tuple[int, ...]] = []
        for vgate in self.virtual_instr():
            if set(vgate.qubits) & self._fragment.qubits:
                conf_list.append(tuple(range(len(vgate.operation.configure()))))
            else:
                conf_list.append((-1,))
        return iter(itertools.product(*conf_list))

    @staticmethod
    def _append_if_in(
        circuit: QuantumCircuit, circ_instr: CircuitInstruction, qubits: Set[Qubit]
    ) -> bool:
        if set(circ_instr.qubits) <= qubits:
            circuit.append(circ_instr)
            return True
        return False

    def _circ_config(self, config_id: Tuple[int, ...]) -> VirtualCircuit:
        conf_reg = ClassicalRegister(len(config_id), name=unique_name())
        conf_circuit = VirtualCircuit(
            *self._vc.qregs.copy(), *self._vc.cregs.copy(), conf_reg
        )

        conf_ctr = 0
        for instr in self._vc.data:
            if isinstance(instr.operation, VirtualBinaryGate):
                if not set(instr.qubits) & self._fragment.qubits:
                    assert config_id[conf_ctr] == -1
                    conf_ctr += 1
                    continue

                elif set(instr.qubits) <= self._fragment.qubits:
                    continue

                elif set(instr.qubits) & self._fragment.qubits:
                    conf_op = instr.operation.configure()[
                        config_id[conf_ctr]
                    ].to_instruction()
                    new_instr = CircuitInstruction(
                        conf_op, instr.qubits, [conf_reg[conf_ctr]]
                    )
                    conf_circuit.data.append(new_instr)
                    conf_ctr += 1
                    continue
                else:
                    raise Exception("unexpected")

            elif set(instr.qubits) <= self._fragment.qubits:
                conf_circuit.append(instr)

        return VirtualCircuit.from_circuit(conf_circuit.decompose()).deflated(
            self._fragment.qubits
        )

    def configured_circuits(self) -> Iterator[Tuple[Tuple[int, ...], VirtualCircuit]]:
        for config_id in self._config_ids():
            yield config_id, self._circ_config(config_id)
