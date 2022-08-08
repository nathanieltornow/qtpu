import itertools
from typing import Dict, List, Optional, Set, Tuple
from qvm.circuit import Circuit
from qvm.circuit.operation import BinaryGate, Measurement, Operation, UnaryGate
from qvm.virtual_circuit.virtual_gate.virtual_gate import Configuration, VirtualGate


class Fragment(Circuit):
    _ids = itertools.count()
    _qubit_map: Dict[int, int]

    def __init__(self, circuit: Circuit, qubits: Set[int]) -> None:
        self.id = next(self._ids)
        qubit_list: List[int] = sorted(qubits)
        self._qubit_map = dict(zip(qubit_list, range(len(qubit_list))))
        super().__init__(
            *self._mapped_operations(circuit._operations),
            num_qubits=circuit.num_clbits,
            num_clbits=circuit.num_clbits,
        )

    @property
    def base_circuit(self) -> Circuit:
        """
        Returns the base circuit of the fragment. The base circuit is the circuit
        without its virtual gates.
        """
        return Circuit(
            *[op for op in self._operations if not isinstance(op, VirtualGate)],
            num_qubits=self.num_qubits,
            num_clbits=self.num_clbits,
        )

    def configured_circuits(self) -> Dict[Tuple[int, ...], Circuit]:
        """
        Creates all the O(6^k) circuit configurations that need to be executed.
        """
        circuit_operations: Dict[Tuple[int, ...], List[Operation]] = {(): []}

        num_clbits_ctr = self._num_clbits
        for op in self._operations:
            if isinstance(op, VirtualGate):
                op.clbit = num_clbits_ctr
                num_clbits_ctr += 1

                new_circuit_operations = {}
                if not (op.qubit1 in self._qubit_map or op.qubit2 in self._qubit_map):
                    for conf_id, circuit_ops in circuit_operations.items():
                        new_circuit_operations[conf_id + (-1,)] = circuit_ops
                else:
                    new_circuit_operations = self._append_configurations(
                        circuit_operations=circuit_operations,
                        configurations=op.configurations(),
                    )
                circuit_operations = new_circuit_operations
            else:
                for ops in circuit_operations.values():
                    ops.append(op)
        return {
            conf_id: Circuit(
                *ops, num_qubits=self._num_qubits, num_clbits=num_clbits_ctr
            )
            for conf_id, ops in circuit_operations.items()
        }

    def _append_configurations(
        self,
        circuit_operations: Dict[Tuple[int, ...], List[Operation]],
        configurations: List[Configuration],
    ) -> Dict[Tuple[int, ...], List[Operation]]:
        new_ops: Dict[Tuple[int, ...], List[Operation]] = {}
        mapped_config = [
            self._mapped_operations(list(config)) for config in configurations
        ]
        for conf_id, ops in circuit_operations.items():
            new_ops.update(
                dict(
                    [
                        (conf_id + (index,), ops + conf)
                        for index, conf in enumerate(mapped_config)
                    ]
                )
            )
        return new_ops

    def _mapped_operations(self, operations: List[Operation]) -> List[Operation]:
        res_operations: List[Operation] = []
        for op in operations:
            if isinstance(op, VirtualGate):
                res_operations.append(op)
            elif isinstance(op, UnaryGate) and op.qubit in self._qubit_map:
                res_operations.append(
                    UnaryGate(
                        name=op.name, qubit=self._qubit_map[op.qubit], params=op.params
                    )
                )
            elif isinstance(op, BinaryGate) and {op.qubit1, op.qubit2}.issubset(
                self._qubit_map
            ):
                res_operations.append(
                    BinaryGate(
                        name=op.name,
                        qubit1=self._qubit_map[op.qubit1],
                        qubit2=self._qubit_map[op.qubit2],
                        params=op.params,
                    )
                )
            elif isinstance(op, Measurement) and op.qubit in self._qubit_map:
                res_operations.append(Measurement(op.qubit, op.clbit))
        return res_operations
