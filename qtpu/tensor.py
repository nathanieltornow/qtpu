import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qtpu.instructions import InstanceGate


def wire_tensor(ind: str) -> qtn.Tensor:
    A = np.array(
        [[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    B = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [-1, -1, 2, 0], [-1, -1, 0, 2]],
        dtype=np.float32,
    )
    return qtn.Tensor(0.5 * A @ B, inds=[f"{ind}_0", f"{ind}_1"], tags=["wire"])


class QuantumTensor:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._instance_gates = [
            instr.operation
            for instr in circuit
            if isinstance(instr.operation, InstanceGate)
        ]
        self._inds = tuple(gate.index for gate in self._instance_gates)
        self._shape = tuple(len(gate.instances) for gate in self._instance_gates)

        self._ind_tensor = qtn.Tensor(
            np.arange(np.prod(self._shape)).reshape(self._shape),
            self._inds,
            tags=["Q"],
        )

        self._instances = None

    @property
    def inds(self) -> tuple[str, ...]:
        return self._inds

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ind_tensor(self) -> qtn.Tensor:
        return self._ind_tensor

    @ind_tensor.setter
    def ind_tensor(self, tensor: qtn.Tensor) -> None:
        self._instances = None
        self._ind_tensor = tensor

    def _flat_ind_to_instance_label(self, flat_ind: int) -> dict[str, int]:
        instance_label = {}
        for ind, size in reversed(list(zip(self._inds, self._shape))):
            instance_label[ind] = flat_ind % size
            flat_ind //= size
        return instance_label

    def get_instance(self, idx: int) -> QuantumCircuit:

        instance_label = self._flat_ind_to_instance_label(idx)

        assert all(label in self._inds for label in instance_label)
        assert len(instance_label) == len(self._inds)

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

    def generate_instances(self) -> list[QuantumCircuit]:
        self._instances = [
            self.get_instance(instance_label)
            for instance_label in self.ind_tensor.data.flat
        ]

    def instances(self) -> list[QuantumCircuit]:
        if self._instances is None:
            self.generate_instances()
        return self._instances


class HybridTensorNetwork:
    def __init__(
        self, quantum_tensors: list[QuantumTensor], qpd_tensors: list[qtn.Tensor]
    ):
        self.quantum_tensors = quantum_tensors
        for i, qt in enumerate(self.quantum_tensors):
            qt.ind_tensor.add_tag(str(i))
        self.qpd_tensors = qpd_tensors

    def simplify(self, tolerance: float = 0.0, max_bond: int = np.inf) -> np.ndarray:
        self.fuse()
        errs = self.approximate(tolerance, max_bond)
        return errs

    def to_tensor_network(self) -> qtn.TensorNetwork:
        return qtn.TensorNetwork(
            [qt.ind_tensor for qt in self.quantum_tensors] + self.qpd_tensors
        )

    def num_circuits(self) -> int:
        return np.sum([qt.ind_tensor.size for qt in self.quantum_tensors])

    def fuse(self) -> None:
        tn = qtn.TensorNetwork([qt.ind_tensor for qt in self.quantum_tensors])
        multibonds = tn.get_multibonds()
        ind_to_qpdtens = {qt.inds[0]: qt for qt in self.qpd_tensors}

        for inds in multibonds.keys():
            tens = qtn.tensor_contract(*[ind_to_qpdtens[ind] for ind in inds])
            for ind in inds:
                ind_to_qpdtens.pop(ind)
            tn.add_tensor(tens)

        for tens in ind_to_qpdtens.values():
            tn.add_tensor(tens)

        tn = tn.fuse_multibonds(inplace=False)

        for qt, new_tens in list(zip(self.quantum_tensors, tn.tensors)):
            qt.ind_tensor = new_tens

        wire_tensors = [tens for tens in self.qpd_tensors if "wire" in tens.tags]
        self.qpd_tensors = list(tn.tensors[len(self.quantum_tensors) :]) + wire_tensors

    def sample(self, num_samples: int = np.inf):
        if num_samples == np.inf:
            return

        ind_to_sort = {}
        all_tensors = [tens for tens in self.qpd_tensors if "wire" in tens.tags]

        errors = []
        for tens in self.qpd_tensors:
            if "wire" in tens.tags:
                continue

            assert len(tens.inds) == 1

            probs = np.abs(tens.data) / np.sum(np.abs(tens.data))
            samples = np.random.choice(len(probs), size=num_samples, p=probs)
            unique, counts = np.unique(samples, return_counts=True)
            indices = unique[counts > 1]

            data = tens.data[indices]

            all_tensors.append(qtn.Tensor(data, inds=tens.inds, tags=tens.tags))
            ind_to_sort[tens.inds[0]] = indices

        for qt in self.quantum_tensors:
            qt.ind_tensor = sort_indices(qt.ind_tensor, ind_to_sort)

        self.qpd_tensors = all_tensors

        return np.array(errors)

    def approximate(self, tolerance: float = 0.0, max_bond: int = np.inf):

        ind_to_sort = {}
        all_tensors = [tens for tens in self.qpd_tensors if "wire" in tens.tags]

        errors = []
        for tens in self.qpd_tensors:
            if "wire" in tens.tags:
                continue

            assert len(tens.inds) == 1

            total = np.sum(np.abs(tens.data))
            sort = np.argsort(np.abs(tens.data))

            norm_data = np.abs(tens.data) / total
            sort = np.argsort(norm_data)
            norm_data = norm_data[sort]

            cumsum = 0.0
            for i in range(len(sort)):
                cumsum += norm_data[i]
                if cumsum > tolerance:
                    break

            cutoff = max(i, len(tens.data) - max_bond)

            errors.append(np.sum(norm_data[:cutoff]))

            data = tens.data[sort[cutoff:]]
            all_tensors.append(qtn.Tensor(data, inds=tens.inds, tags=tens.tags))
            ind_to_sort[tens.inds[0]] = sort[cutoff:]

        for qt in self.quantum_tensors:
            qt.ind_tensor = sort_indices(qt.ind_tensor, ind_to_sort)

        self.qpd_tensors = all_tensors

        return np.array(errors)

    def contraction_cost(self) -> int:
        return self.to_tensor_network().contraction_cost(optimize="auto-hq")

    def draw(self, color=["QPD", "Q", "wire"], **kwargs):
        return self.to_tensor_network().draw(color=color, **kwargs)


def sort_indices(tensor: qtn.Tensor, ind_to_sort: dict[str, np.ndarray]) -> qtn.Tensor:
    data = tensor.data
    for ind, sort in ind_to_sort.items():
        if ind not in tensor.inds:
            continue
        dim = tensor.inds.index(ind)
        data = np.take(data, sort, axis=dim)
    return qtn.Tensor(data, inds=tensor.inds, tags=tensor.tags)
