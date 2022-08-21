import itertools
from typing import Any, Dict, Iterator, List, Optional, Tuple

import lithops.multiprocessing as mp

from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit import transpile

from qvm.circuit.virtual_circuit import VirtualBinaryGate
from qvm.circuit import VirtualCircuit
from qvm.result import Result
from .knit import knit

from qvm import util

DEFAULT_TRANSPILER_FLAGS = {"optimization_level": 3}

DEFAULT_EXEC_FLAGS = {"shots": 10000}


class VirtualCircuitExecutor:
    _vc: VirtualCircuit
    _pool: Optional[mp.Pool] = None

    def __init__(self, virtual_circuit: VirtualCircuit, pool: Optional[mp.Pool] = None):
        self._vc = virtual_circuit
        self._pool = pool

    def _virtual_gates(self) -> List[VirtualBinaryGate]:
        return [
            instr.operation
            for instr in self._vc.circuit().data
            if isinstance(instr.operation, VirtualBinaryGate)
        ]

    def _config_ids(self) -> Iterator[Tuple[int, ...]]:
        conf_list: List[Tuple[int, ...]] = [
            tuple(range(len(vgate.configure()))) for vgate in self._virtual_gates()
        ]
        return iter(itertools.product(*conf_list))

    def _configured_circuits(self) -> Iterator[Tuple[Tuple[int, ...], QuantumCircuit]]:
        for config_id in self._config_ids():
            yield config_id, self._circuit_with_config(config_id)

    def _circuit_with_config(self, conf_id: Tuple[int, ...]) -> QuantumCircuit:
        orig_circuit = self._vc.circuit()

        if len(conf_id) != len(self._virtual_gates()):
            raise ValueError("config length does not match virtual gate length")
        if len(conf_id) == 0:
            return orig_circuit

        num_clbits = orig_circuit.num_clbits + len(conf_id)
        configured_circuit = QuantumCircuit(orig_circuit.num_qubits, num_clbits)

        conf_index = 0
        for circ_instr in orig_circuit.data:
            if isinstance(circ_instr.operation, VirtualBinaryGate):
                instruction = circ_instr.operation.configure()[
                    conf_id[conf_index]
                ].to_instruction(label=f"config{conf_id[conf_index]}")
                clbit = configured_circuit.clbits[orig_circuit.num_clbits + conf_index]
                new_instr = CircuitInstruction(
                    instruction,
                    util.mapped_qubits(
                        orig_circuit, configured_circuit, circ_instr.qubits
                    ),
                    [clbit],
                )
                configured_circuit.append(new_instr)
                conf_index += 1
            else:
                configured_circuit.append(
                    util.mapped_instruction(
                        orig_circuit, configured_circuit, circ_instr
                    )
                )
        return configured_circuit

    def _knit(self, results: Dict[Tuple[int, ...], Result]) -> Result:
        return knit(results, self._virtual_gates())

    def execute(
        self,
        backend: Backend,
        transpile_flags: Dict[str, Any] = DEFAULT_TRANSPILER_FLAGS,
        exec_flags: Dict[str, Any] = DEFAULT_TRANSPILER_FLAGS,
    ) -> Result:
        # if the circuit is not virtualized, just execute it
        if len(self._virtual_gates()) == 0:
            print(self._vc.circuit())
            t_circ = transpile(self._vc.circuit(), backend, **transpile_flags)
            return Result.from_counts(
                backend.run(t_circ, **exec_flags).result().get_counts()
            )

        conf_ids, conf_circs = zip(*self._configured_circuits())
        conf_ids = list(conf_ids)
        conf_circs = list(conf_circs)

        t_circs = transpile(conf_circs, backend, **transpile_flags)
        results = [
            Result.from_counts(cnt)
            for cnt in backend.run(t_circs, **exec_flags).result().get_counts()
        ]
        return self._knit(dict(zip(conf_ids, results)))
