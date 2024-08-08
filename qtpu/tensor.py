import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qtpu.instructions import InstanceGate
from qtpu.helpers import defer_mid_measurements


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

    # setter for ind_tensor
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

        res_circuit = res_circuit.decompose()
        res_circuit = defer_mid_measurements(res_circuit)
        return res_circuit

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

    def approximate(self, tolerance: float = 0.01) -> None:
        tn = qtn.TensorNetwork([qt.ind_tensor for qt in self.quantum_tensors])
        multibonds = tn.get_multibonds()
        tn.fuse_multibonds(inplace=True)

        new_qpds = [qpd for qpd in self.qpd_tensors if "wire" in qpd.tags]
        ind_to_qpdtens = {qt.inds[0]: qt for qt in self.qpd_tensors}

        ind_to_argsort = {}
        for mb_inds in multibonds.keys():
            qpd_tensors = [ind_to_qpdtens[ind] for ind in mb_inds]
            qpd_kron = np.array([1])
            for qpd in qpd_tensors:
                qpd_kron = np.kron(qpd_kron, qpd.data)

            sort = np.argsort(np.abs(qpd_kron))
            qpd_kron = qpd_kron[sort]

            # get the index until the cumulative sum is less than tolerance
            s = 0
            for i, val in enumerate(qpd_kron):
                s += val
                if s > tolerance:
                    break

            qpd_kron = qpd_kron[i:]

            ind_to_argsort[mb_inds[0]] = sort[i:]

            tens = qtn.Tensor(qpd_kron, [mb_inds[0]], tags=["QPD"])
            new_qpds.append(tens)

        self.qpd_tensors = new_qpds

        for i, qt in enumerate(self.quantum_tensors):
            qt.ind_tensor = sort_indices(tn.tensors[i], ind_to_argsort)
            assert str(i) in qt.ind_tensor.tags

    def to_tensor_network(self) -> qtn.TensorNetwork:
        return qtn.TensorNetwork(
            [qt.ind_tensor for qt in self.quantum_tensors] + self.qpd_tensors
        )


def sort_indices(tensor: qtn.Tensor, ind_to_sort: dict[str, np.ndarray]) -> qtn.Tensor:
    data = tensor.data
    for ind, sort in ind_to_sort.items():
        if ind not in tensor.inds:
            continue
        dim = tensor.inds.index(ind)
        data = np.take(data, sort, axis=dim)
    return qtn.Tensor(data, inds=tensor.inds, tags=tensor.tags)


def truncate_tensor_index(tensor: qtn.Tensor, ind: str, new_size: int) -> qtn.Tensor:
    if ind not in tensor.inds:
        return tensor
    dim = tensor.inds.index(ind)
    slices = [slice(None)] * len(tensor.shape)
    slices[dim] = slice(new_size)
    truncated_data = tensor.data[tuple(slices)]
    return qtn.Tensor(truncated_data, inds=tensor.inds, tags=tensor.tags)


def truncate_tn_index(
    tn: qtn.TensorNetwork, ind: str, new_size: int
) -> qtn.TensorNetwork:
    return qtn.TensorNetwork(
        [truncate_tensor_index(t, ind, new_size) for t in tn.tensors]
    )


def find_fusable_indices(tn: qtn.TensorNetwork) -> list[set[str]]:
    # find the groups of indices that have the same set of tensors with tags "Q"
    groups: dict[tuple[int, ...], list[str]] = {}

    for index, tids in tn.ind_map.items():
        q_tids = tuple(sorted([tid for tid in tids if "Q" in tn.tensor_map[tid].tags]))
        groups.setdefault(q_tids, []).append(index)

    return groups
