import itertools
from typing import Any, Dict, Iterator, List, Optional, Tuple

from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit, CircuitInstruction

from qvm.circuit.virtual_gate.virtual_gate import PartialVirtualGate, VirtualBinaryGate
from qvm.execution.knit import knit
from qvm.execution.merge import merge
from qvm.result import Result
from qvm.circuit import VirtualCircuit, FragmentedVirtualCircuit, Fragment
from qvm import util
from .executor import (
    DEFAULT_TRANSPILER_FLAGS,
    DEFAULT_EXEC_FLAGS,
    VirtualCircuitExecutor,
)


class FragmentedCircuitExecutor:
    _fvc: FragmentedVirtualCircuit

    def __init__(self, fragmented_circuit: FragmentedVirtualCircuit) -> None:
        self._fvc = fragmented_circuit

    def _virtual_gates(self) -> List[VirtualBinaryGate]:
        """
        Returns the virtual gates that are between the fragments in fvc.
        """
        orig_circ = self._fvc.circuit()
        res_virtual_gates: List[VirtualBinaryGate] = []
        for instr in orig_circ.data:
            if isinstance(instr.operation, VirtualBinaryGate):
                instr_qubit_indices = set(util.bit_indices(orig_circ, instr.qubits))
                if not any(
                    instr_qubit_indices <= frag.qubit_indices()
                    for frag in self._fvc.fragments()
                ):
                    res_virtual_gates.append(instr.operation)
        return res_virtual_gates

    def _config_ids(self, fragment: Fragment) -> Iterator[Tuple[int, ...]]:
        conf_id_list: List[List[int]] = []
        orig_circ = self._fvc.circuit()
        virtual_gates = self._virtual_gates()
        for instr in orig_circ.data:
            if (
                isinstance(instr.operation, VirtualBinaryGate)
                and instr.operation in virtual_gates
            ):
                if (
                    orig_circ.find_bit(instr.qubits[0]).index
                    in fragment.qubit_indices()
                    or orig_circ.find_bit(instr.qubits[1]).index
                    in fragment.qubit_indices()
                ):
                    conf_id_list.append(list(range(len(instr.operation.configure()))))
                else:
                    conf_id_list.append([-1])
        return iter(itertools.product(*conf_id_list))

    def _circuit_with_configuration(
        self, fragment: Fragment, config_id: Tuple[int, ...]
    ) -> VirtualCircuit:
        orig_circ = self._fvc.circuit()
        frag_qubits = {orig_circ.qubits[i] for i in fragment.qubit_indices()}

        new_circuit = QuantumCircuit(
            orig_circ.num_qubits, orig_circ.num_clbits + len(config_id)
        )

        conf_ctr = 0
        for instr in orig_circ.data:
            if not isinstance(instr.operation, VirtualBinaryGate):
                if set(instr.qubits) <= frag_qubits:
                    new_circuit.append(
                        util.mapped_instruction(orig_circ, new_circuit, instr)
                    )
                continue

            if instr not in self._virtual_gates():
                continue

            if config_id[conf_ctr] == -1:
                conf_ctr += 1
                continue

            # if the instruction has to be handled by the fragmented circuit
            if (instr.qubits[0] in frag_qubits) ^ (instr.qubits[1] in frag_qubits):
                qubit_in_fragment = (
                    instr.qubits[0]
                    if instr.qubit[0] in frag_qubits
                    else instr.qubits[1]
                )
                conf_circ = instr.operation.configure()[config_id[conf_ctr]]
                conf_circ = util.circuit_on_qubits(
                    conf_circ, {qubit_in_fragment}, deflated=True
                )
                conf_instr = conf_circ.to_instruction()
                clbit = orig_circ.clbits[orig_circ.num_clbits + conf_ctr]
                new_circuit.append(
                    conf_instr,
                    util.mapped_qubits(orig_circ, new_circuit, instr.qubits),
                    [clbit],
                )
                conf_ctr += 1
            else:
                raise Exception("This should not happen")

        return VirtualCircuit(new_circuit).deflated()

    def _configured_fragment(
        self, fragment: Fragment
    ) -> Iterator[Tuple[Tuple[int, ...], VirtualCircuit]]:
        for config_id in self._config_ids(fragment):
            yield config_id, self._circuit_with_configuration(fragment, config_id)

    def _execute_fragment(
        self,
        fragment: Fragment,
        backend: Backend,
        transpile_flags: Dict[str, Any] = DEFAULT_TRANSPILER_FLAGS,
        exec_flags: Dict[str, Any] = DEFAULT_EXEC_FLAGS,
    ) -> Dict[Tuple[int, ...], Result]:
        res_dict: Dict[Tuple[int, ...], Result] = {}
        for config_id, vc in self._configured_fragment(fragment):
            executor = VirtualCircuitExecutor(vc)
            res = executor.execute(backend, transpile_flags, exec_flags)
            res_dict[config_id] = res
        return res_dict

    def execute(
        self,
        backend: Backend,
        transpile_flags: Dict[str, Any] = DEFAULT_TRANSPILER_FLAGS,
        exec_flags: Dict[str, Any] = DEFAULT_EXEC_FLAGS,
    ) -> Result:
        """
        Executes the fragmented circuit.
        """
        frag_res = []
        for frag in self._fvc.fragments():
            frag_res.append(
                self._execute_fragment(frag, backend, transpile_flags, exec_flags)
            )
        merged = merge(frag_res)
        return knit(merged, self._virtual_gates())
