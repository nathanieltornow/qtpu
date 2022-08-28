from dataclasses import dataclass
import itertools
import secrets
from typing import Dict, Iterator, List, Tuple

from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister

from qvm.transpiler.fragmented_circuit import FragmentedCircuit, Fragment
from qvm.virtual_gate import VirtualBinaryGate, VirtualGateEndpoint


def unique_name() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(secrets.choice(alphabet) for i in range(4))


@dataclass
class VirtualGateInfo:
    vgate: VirtualBinaryGate
    # fragment and the operation index in the fragment
    endpoint0: Tuple[Fragment, int]
    endpoint1: Tuple[Fragment, int]


def compute_virtual_gate_info(
    frag_circuit: FragmentedCircuit,
) -> List[VirtualGateInfo]:
    vgates: List[VirtualGateInfo] = []
    pending: Dict[
        VirtualBinaryGate,
        Tuple[Fragment, int],
    ] = {}
    for frag in frag_circuit.fragments:
        for index, circ_instr in enumerate(frag.data):
            if isinstance(circ_instr.operation, VirtualBinaryGate):
                raise ValueError(
                    "Virtual gates should not be in the circuit, please decompose first"
                )
            if isinstance(circ_instr.operation, VirtualGateEndpoint):
                endpoint = circ_instr.operation
                if endpoint.gate in pending:
                    first = (
                        (frag, index) if endpoint.index == 0 else pending[endpoint.gate]
                    )
                    second = (
                        (frag, index) if endpoint.index == 1 else pending[endpoint.gate]
                    )
                    vgates.append(VirtualGateInfo(endpoint.gate, first, second))
                    del pending[endpoint.gate]
                else:
                    pending[endpoint.gate] = (frag, index)
    if len(pending) > 0:
        raise ValueError("Some virtual gates are not paired")
    return vgates


class Configurator:
    _fragment: Fragment
    _vgates: List[VirtualGateInfo]

    def __init__(self, fragment: Fragment, vgates: List[VirtualGateInfo]):
        self._fragment = fragment
        self._vgates = vgates

    def _config_ids(self) -> Iterator[Tuple[int, ...]]:
        conf_list = []
        for vg in self._vgates:
            if self._fragment in {vg.endpoint0[0], vg.endpoint1[0]}:
                conf_list.append(tuple(range(len(vg.vgate.configure()))))
            else:
                conf_list.append((-1,))
        return iter(itertools.product(*conf_list))

    def _circuit_with_config(self, config_id: Tuple[int, ...]) -> QuantumCircuit:
        config_circ = self._fragment.copy()
        config_register = ClassicalRegister(len(config_id), name=unique_name())
        config_circ.add_register(config_register)

        for i, (vgate_info, conf) in enumerate(zip(self._vgates, config_id)):
            if conf == -1:
                continue
            clbits = (config_register[i],)
            frag0, index0 = vgate_info.endpoint0
            frag1, index1 = vgate_info.endpoint1
            config0, config1 = vgate_info.vgate.partial_config(conf)

            if frag0 == self._fragment:
                config_circ.data[index0] = CircuitInstruction(
                    config0.to_instruction(), config_circ.data[index0].qubits, clbits
                )
            if frag1 == self._fragment:
                config_circ.data[index1] = CircuitInstruction(
                    config1.to_instruction(), config_circ.data[index1].qubits, clbits
                )
        return config_circ

    def configured_circuits(self) -> Iterator[Tuple[Tuple[int, ...], QuantumCircuit]]:
        for conf_id in self._config_ids():
            yield conf_id, self._circuit_with_config(conf_id)


class VirtualizationInfo:
    _virtual_gates: List[VirtualBinaryGate]

    def __init__(self, fragmented_circuit: FragmentedCircuit) -> None:
        self._virtual_gates = [
            info.vgate for info in compute_virtual_gate_info(fragmented_circuit)
        ]

    @property
    def virtual_gates(self) -> List[VirtualBinaryGate]:
        return self._virtual_gates

    @property
    def config_ids(self) -> Iterator[Tuple[int, ...]]:
        conf_list = [tuple(range(len(vg.configure()))) for vg in self.virtual_gates]
        return iter(itertools.product(*conf_list))
