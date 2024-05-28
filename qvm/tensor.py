import itertools
from typing import Iterator
import numpy as np
from numpy.typing import NDArray

from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qvm.instructions import InstanceGate


class ClassicalTensor:
    def __init__(self, data: NDArray, inds: tuple[str, ...]) -> None:
        assert len(data.shape) == len(inds)
        self._data = data
        self._inds = inds

    @property
    def data(self) -> NDArray:
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def inds(self) -> tuple[chr, ...]:
        return self._inds


class QuantumTensor:
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._instance_gates = [
            instr.operation
            for instr in circuit
            if isinstance(instr.operation, InstanceGate)
        ]
        self._indices = tuple(gate.index for gate in self._instance_gates)
        self._shape = tuple(len(gate.instances) for gate in self._instance_gates)

        self._shot_portions = np.array([1])

        if len(self._instance_gates) == 0:
            return

        self._shot_portions = np.array(self._instance_gates[0].shot_portion)
        for gate in self._instance_gates[1:]:
            self._shot_portions = np.kron(self._shot_portions, gate.shot_portion)

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def inds(self) -> tuple[str]:
        return self._indices

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def instances(self) -> Iterator[tuple[QuantumCircuit, float]]:
        for instance_label, shot_portion in self._instance_labels():
            yield self._get_instance(instance_label), shot_portion

    def _instance_labels(self) -> Iterator[tuple[dict[str, int], float]]:
        for instance_label, shot_portion in zip(
            itertools.product(*[range(n) for n in self._shape]), self._shot_portions
        ):
            yield dict(zip(self._indices, instance_label)), shot_portion

    def _get_instance(self, instance_label: dict[str, int]) -> QuantumCircuit:
        assert all(label in self._indices for label in instance_label)
        assert len(instance_label) == len(self._indices)

        res_circuit = QuantumCircuit(*self._circuit.qregs, *self._circuit.cregs)

        for instr in self._circuit:
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
            if isinstance(op, InstanceGate):
                inst_cnt = instance_label[op.label]
                inst_circuit = op.instances[inst_cnt]

                if len(qubits) != inst_circuit.num_qubits:
                    raise ValueError(
                        f"Parameterized gate {op.index} requires {inst_circuit.num_qubits} qubits, got {len(qubits)}"
                    )

                creg = ClassicalRegister(
                    size=inst_circuit.num_clbits, name=f"c_{op.index}"
                )
                res_circuit.add_register(creg)

                op = inst_circuit.to_instruction()
                clbits = [creg[i] for i in range(inst_circuit.num_clbits)]

            res_circuit.append(op, qubits, clbits)

        return res_circuit.decompose()


class HybridTensorNetwork:
    def __init__(
        self, quantum_tensors: list[QuantumTensor], classical_tensors: list[ClassicalTensor]
    ) -> None:
        self._quantum_tensors = quantum_tensors
        self._classical_tensors = classical_tensors

        self._index_map: dict[str, tuple[set[QuantumTensor | ClassicalTensor], int]] = {}

        self._index_to_chr = {
            index: chr(65 + i)
            for i, index in enumerate(
                itertools.chain.from_iterable(
                    tens.inds for tens in quantum_tensors + classical_tensors
                )
            )
        }
        assert len(self._index_to_chr) < 0x11000

        for tensor in quantum_tensors + classical_tensors:
            for index, index_size in zip(tensor.inds, tensor.shape):
                if index not in self._index_map:
                    self._index_map[index] = (set(), index_size)

                if len(self._index_map[index][0]) > 1:
                    raise ValueError(
                        f"Index {index} is already used by more than one tensor"
                    )
                if index_size != self._index_map[index][1]:
                    raise ValueError(
                        f"Index {index} has different size than other tensors"
                    )

                self._index_map[index][0].add(tensor)

    @property
    def quantum_tensors(self) -> list[QuantumTensor]:
        return self._quantum_tensors.copy()

    @property
    def classical_tensors(self) -> list[ClassicalTensor]:
        return self._classical_tensors.copy()

    def inputs(self) -> list[tuple[str, ...]]:
        return [
            tuple(self._index_to_chr[ind] for ind in tens.inds)
            for tens in self._quantum_tensors + self._classical_tensors
        ]

    def output(self) -> tuple[str, ...]:
        return tuple(
            [
                self._index_to_chr[index]
                for index, tensors in self._index_map.items()
                if len(tensors[0]) == 1
            ]
        )

    def size_dict(self) -> dict[str, int]:
        return {
            self._index_to_chr[index]: size
            for index, (_, size) in self._index_map.items()
        }

    def equation(self) -> str:
        in_str = ("".join(ix for ix in inds) for inds in self.inputs())
        in_str = ",".join(in_str)
        out_str = "".join(
            ix for ix in self._index_to_chr.values() if in_str.count(ix) == 1
        )
        return f"{in_str}->{out_str}"
