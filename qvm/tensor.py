import itertools
from typing import Iterator

import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit, Barrier, ClassicalRegister


class InstanceGate(Barrier):
    def __init__(self, num_qubits: int, index: str, instances: list[QuantumCircuit]):
        assert all(inst.num_qubits == num_qubits for inst in instances)

        self._index = index
        self._instances = instances
        super().__init__(num_qubits, label=index)

    @property
    def index(self) -> str:
        return self._index

    @property
    def instances(self) -> list[QuantumCircuit]:
        return self._instances


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

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def inds(self) -> tuple[str]:
        return self._indices

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def instances(self) -> Iterator[QuantumCircuit]:
        for instance_label in self._instance_labels():
            yield self._get_instance(instance_label)

    def _instance_labels(self) -> Iterator[dict[str, int]]:
        for instance_label in itertools.product(*[range(n) for n in self._shape]):
            yield dict(zip(self._indices, instance_label))

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
        self, quantum_tensors: list[QuantumTensor], classical_tensors: list[qtn.Tensor]
    ) -> None:
        self._quantum_tensors = quantum_tensors
        self._classical_tensors = classical_tensors

        self._index_map: dict[str, tuple[set[QuantumTensor | qtn.Tensor], int]] = {}

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
    def classical_tensors(self) -> list[qtn.Tensor]:
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
