""""A module to represent quantum-circuit tensors and hybrid tensor networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from qiskit.circuit import Instruction, Parameter, QuantumCircuit

if TYPE_CHECKING:
    from numpy.typing import NDArray


class InstructionVector(Instruction):  # type: ignore[misc]
    """InstructionVector class represents a vector of one-qubit quantum instructions.

    Attributes:
    ----------
        param (Parameter): The parameter used to index the instruction vector.
        vector (list[list[Instruction]]): The vector of quantum instructions.
    """

    def __init__(
        self,
        instructions_vector: list[list[Instruction]],
        idx_param: Parameter,
        weights: NDArray[np.float32] | None = None,
    ) -> None:
        """Initialize the InstructionVector object with a vector of quantum instructions.

        Args:
            instructions_vector (list[list[Instruction]]): A vector of quantum instructions.
            idx_param (Parameter): The parameter used to index the instruction vector.
            weights (list[float], optional): A list of weights for each instruction in the vector.
                Defaults to None.
        """
        assert all(
            instr.num_qubits == 1 for instrs in instructions_vector for instr in instrs
        )
        assert isinstance(idx_param, Parameter)

        super().__init__("vec", 1, 0, params=(idx_param,))
        self._vector = instructions_vector

        if weights is None:
            weights = np.ones(len(self), dtype=np.float32)

        assert weights.shape == (len(instructions_vector),)
        self._weights = weights / np.sum(weights)

    @property
    def param(self) -> Parameter:
        """Returns the parameter used to index the instruction vector.

        Returns:
            Parameter: The parameter used to index the instruction vector.
        """
        return self.params[0]

    @property
    def vector(self) -> list[list[Instruction]]:
        """Returns the vector of quantum instructions.

        Returns:
            list[list[Instruction]]: The vector of quantum instructions.
        """
        return self._vector

    @property
    def weights(self) -> NDArray[np.float32]:
        """Returns the weights of the instructions in the vector.

        Returns:
            NDArray[np.float32]: The weights of the instructions in the vector.
        """
        return self._weights

    def __len__(self) -> int:
        """Returns the length of the instruction vector.

        Returns:
            int: The length of the instruction vector.
        """
        return len(self.vector)

    def _define(self) -> None:
        circuit = QuantumCircuit(1, 0)
        param_value = int(self.params[0])

        for instr in self.vector[param_value]:
            circuit.append(instr, [0])
        self.definition = circuit


class CircuitTensor:
    """A class to represent a tensor of quantum circuits.

    The tensor property comes from having InstructionVector operations in a quantum circuit.
    The circuit tensor is then the tensor product of all possible circuits that can be obtained by assigning
    different values to the parameters of the InstructionVector operations.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit associated with the tensor.

    Attributes:
    ----------
    circuit : QuantumCircuit
        The quantum circuit associated with the tensor.
    vector_ops : list[InstructionVector]
        A list of InstructionVector operations in the circuit.
    shape : tuple[int, ...]
        The shape of the tensor, representing the number of parameters for each operation.
    inds : tuple[str, ...]
        The names of the parameters in the tensor.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        """Initialize the tensor with a given quantum circuit.

        Args:
            circuit (QuantumCircuit): The quantum circuit to initialize the tensor with.
        """
        param_to_op = list(
            {
                instr.operation.param: instr.operation
                for instr in circuit
                if isinstance(instr.operation, InstructionVector)
            }.items()
        )

        self._circuit = circuit
        self._vector_ops = [op for _, op in param_to_op]
        self._shape: tuple[int, ...] = tuple(len(op) for _, op in param_to_op)
        self._inds: tuple[str, ...] = tuple(p.name for p, _ in param_to_op)

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the QuantumCircuit object associated with this instance.

        Returns:
            QuantumCircuit: The quantum circuit associated with this instance.
        """
        return self._circuit

    @property
    def vector_ops(self) -> list[InstructionVector]:
        """Returns the InstructionVector operations in the circuit.

        Returns:
            list[InstructionVector]: A list of InstructionVector operations in the circuit.
        """
        return self._vector_ops

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the tensor.

        Returns:
            tuple[int, ...]: A tuple representing the dimensions of the tensor.
        """
        return self._shape

    @property
    def inds(self) -> tuple[str, ...]:
        """Returns the indices of the tensor (quimb-style).

        Returns:
            tuple[str, ...]: A tuple of strings representing the indices of the tensor.
        """
        return self._inds

    def __getitem__(self, index: int | tuple[int, ...]) -> QuantumCircuit:
        """Returns the circuit corresponding to the provided index.

        Args:
            index (int | tuple[int, ...]): The index of the circuit to be returned.

        Returns:
            QuantumCircuit: The circuit corresponding to the provided index.
        """
        if isinstance(index, int):
            index = (index,)

        assert len(index) == len(self.shape)

        param_map = dict(zip(self.inds, index, strict=False))
        # TODO: might be a unnecessary copy
        return self._circuit.copy().assign_parameters(param_map)

    def param_assignment(self, key: int | tuple[int, ...]) -> dict[str, int]:
        """Assigns parameters based on the provided key.

        Args:
            key (tuple[int, ...]): A tuple of integers representing the indices to be assigned.
                                   If a single integer is provided, it will be converted to a tuple.

        Returns:
            dict[str, int]: A dictionary mapping each index name (as a string) to its corresponding value from the key.

        Raises:
            AssertionError: If the length of the key does not match the length of the shape attribute.
        """
        if isinstance(key, int):
            key = (key,)

        assert len(key) == len(self.shape)

        return dict(zip(self.inds, key, strict=False))

    def get_param_assignments(self) -> list[dict[str, int]]:
        """Get parameter assignments for all indices in the tensor's shape.

        Returns:
            list[dict[str, int]]: A list of dictionaries where each dictionary represents
                                  the parameter assignments for a specific index in the tensor.
        """
        indices = np.ndindex(self.shape)
        return [self.param_assignment(tuple(idx)) for idx in indices]

    def flat(self) -> list[QuantumCircuit]:
        """Returns the circuit tensor as a flat list.

        Returns:
            list[QuantumCircuit]: The circuit tensor flattened into a list.
        """
        indices = np.ndindex(self.shape)
        return [self[tuple(idx)] for idx in indices]

    def fuse(self, indices: list[str]) -> None:
        """Fuse the tensor-indices along the specified indices.

        Parameters:
        indices (list[str]): A list of indices to fuse.

        Raises:
        NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError


class HybridTensorNetwork:
    """A class to represent a hybrid tensor network consisting of quantum and classical tensors.

    Attributes:
    -----------
    qtensors : list[CircuitTensor]
        A list of quantum circuit tensors.
    ctensors : list[qtn.Tensor]
        A list of classical tensors.
    """

    def __init__(
        self, qtensors: list[CircuitTensor], ctensors: list[qtn.Tensor]
    ) -> None:
        """Initialize the tensor object with quantum and classical tensors.

        Args:
            qtensors (list[CircuitTensor]): A list of quantum circuit tensors.
            ctensors (list[qtn.Tensor]): A list of classical tensors.

        Returns:
            None.
        """
        self._qts = qtensors
        self._cts = ctensors

    @property
    def qtensors(self) -> list[CircuitTensor]:
        """Returns the quantum tensors in the tensor network.

        Returns:
            list[CircuitTensor]: A list of quantum tensors in the tensor network.
        """
        return self._qts

    @property
    def ctensors(self) -> list[qtn.Tensor]:
        """Returns the classical tensors in the tensor network.

        Returns:
            list[qtn.Tensor]: A list of classical tensors in the tensor network.
        """
        return self._cts

    @property
    def subcircuits(self) -> list[QuantumCircuit]:
        """Returns the circuits corresponding to the quantum tensors.

        Returns:
            list[QuantumCircuit]: A list of quantum circuits corresponding to the quantum tensors.
        """
        return [qt.circuit for qt in self._qts]

    def to_dummy_tn(self) -> qtn.TensorNetwork:
        """Convert the current object to a dummy classical tensor network.

        To be used for analysis purposes.

        Returns:
            qtn.TensorNetwork: A tensor network consisting of dummy quantum tensors
                and the original classical tensors.
        """
        qt_dummys = [
            qtn.Tensor(np.zeros(qt.shape), inds=qt.inds, tags=["Q"]) for qt in self._qts
        ]
        return qtn.TensorNetwork(qt_dummys + self._cts)


# class QuantumTensor:
#     def __init__(self, circuit: QuantumCircuit) -> None:
#         self._circuit = circuit
#         self._instance_gates = [
#             instr.operation
#             for instr in circuit
#             if isinstance(instr.operation, InstanceGate)
#         ]
#         self._inds = tuple(gate.index for gate in self._instance_gates)
#         self._shape = tuple(len(gate.instances) for gate in self._instance_gates)

#         self._ind_tensor = qtn.Tensor(
#             np.arange(np.prod(self._shape)).reshape(self._shape),
#             self._inds,
#             tags=["Q"],
#         )

#         self._instances = None

#     @property
#     def inds(self) -> tuple[str, ...]:
#         return self._inds

#     @property
#     def shape(self) -> tuple[int, ...]:
#         return self._shape

#     @property
#     def ind_tensor(self) -> qtn.Tensor:
#         return self._ind_tensor

#     @ind_tensor.setter
#     def ind_tensor(self, tensor: qtn.Tensor) -> None:
#         self._instances = None
#         self._ind_tensor = tensor

#     def _flat_ind_to_instance_label(self, flat_ind: int) -> dict[str, int]:
#         instance_label = {}
#         for ind, size in reversed(list(zip(self._inds, self._shape))):
#             instance_label[ind] = flat_ind % size
#             flat_ind //= size
#         return instance_label

#     def get_instance(self, idx: int) -> QuantumCircuit:

#         instance_label = self._flat_ind_to_instance_label(idx)

#         assert all(label in self._inds for label in instance_label)
#         assert len(instance_label) == len(self._inds)

#         res_circuit = QuantumCircuit(*self._circuit.qregs, *self._circuit.cregs)

#         for instr in self._circuit:
#             op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
#             if isinstance(op, InstanceGate):
#                 inst_cnt = instance_label[op.label]
#                 inst_circuit = op.instances[inst_cnt]

#                 if len(qubits) != inst_circuit.num_qubits:
#                     raise ValueError(
#                         f"Parameterized gate {op.index} requires {inst_circuit.num_qubits} qubits, got {len(qubits)}"
#                     )

#                 creg = ClassicalRegister(
#                     size=inst_circuit.num_clbits, name=f"c_{op.index}"
#                 )
#                 res_circuit.add_register(creg)

#                 op = inst_circuit.to_instruction()
#                 clbits = [creg[i] for i in range(inst_circuit.num_clbits)]

#             res_circuit.append(op, qubits, clbits)

#         return res_circuit.decompose()

#     def generate_instances(self) -> list[QuantumCircuit]:
#         self._instances = [
#             self.get_instance(instance_label)
#             for instance_label in self.ind_tensor.data.flat
#         ]

#     def instances(self) -> list[QuantumCircuit]:
#         if self._instances is None:
#             self.generate_instances()
#         return self._instances


# class HybridTensorNetwork:
#     def __init__(
#         self, quantum_tensors: list[QuantumTensor], qpd_tensors: list[qtn.Tensor]
#     ):
#         self.quantum_tensors = quantum_tensors
#         for i, qt in enumerate(self.quantum_tensors):
#             qt.ind_tensor.add_tag(str(i))
#         self.qpd_tensors = qpd_tensors

#     def simplify(self, tolerance: float = 0.0, max_bond: int = np.inf) -> np.ndarray:
#         self.fuse()
#         errs = self.approximate(tolerance, max_bond)
#         return errs

#     def to_tensor_network(self) -> qtn.TensorNetwork:
#         return qtn.TensorNetwork(
#             [qt.ind_tensor for qt in self.quantum_tensors] + self.qpd_tensors
#         )

#     def num_circuits(self) -> int:
#         return np.sum([qt.ind_tensor.size for qt in self.quantum_tensors])

#     def fuse(self) -> None:
#         tn = qtn.TensorNetwork([qt.ind_tensor for qt in self.quantum_tensors])
#         multibonds = tn.get_multibonds()
#         ind_to_qpdtens = {qt.inds[0]: qt for qt in self.qpd_tensors}

#         for inds in multibonds.keys():
#             tens = qtn.tensor_contract(*[ind_to_qpdtens[ind] for ind in inds])
#             for ind in inds:
#                 ind_to_qpdtens.pop(ind)
#             tn.add_tensor(tens)

#         for tens in ind_to_qpdtens.values():
#             tn.add_tensor(tens)

#         tn = tn.fuse_multibonds(inplace=False)

#         for qt, new_tens in list(zip(self.quantum_tensors, tn.tensors)):
#             qt.ind_tensor = new_tens

#         wire_tensors = [tens for tens in self.qpd_tensors if "wire" in tens.tags]
#         self.qpd_tensors = list(tn.tensors[len(self.quantum_tensors) :]) + wire_tensors

#     def sample(self, num_samples: int = np.inf):
#         if num_samples == np.inf:
#             return

#         ind_to_sort = {}
#         all_tensors = [tens for tens in self.qpd_tensors if "wire" in tens.tags]

#         errors = []
#         for tens in self.qpd_tensors:
#             if "wire" in tens.tags:
#                 continue

#             assert len(tens.inds) == 1

#             probs = np.abs(tens.data) / np.sum(np.abs(tens.data))
#             samples = np.random.choice(len(probs), size=num_samples, p=probs)
#             unique, counts = np.unique(samples, return_counts=True)
#             indices = unique[counts > 1]

#             data = tens.data[indices]

#             all_tensors.append(qtn.Tensor(data, inds=tens.inds, tags=tens.tags))
#             ind_to_sort[tens.inds[0]] = indices

#         for qt in self.quantum_tensors:
#             qt.ind_tensor = sort_indices(qt.ind_tensor, ind_to_sort)

#         self.qpd_tensors = all_tensors

#         return np.array(errors)

#     def approximate(self, tolerance: float = 0.0, max_bond: int = np.inf):

#         ind_to_sort = {}
#         all_tensors = [tens for tens in self.qpd_tensors if "wire" in tens.tags]

#         errors = []
#         for tens in self.qpd_tensors:
#             if "wire" in tens.tags:
#                 continue

#             assert len(tens.inds) == 1

#             total = np.sum(np.abs(tens.data))
#             sort = np.argsort(np.abs(tens.data))

#             norm_data = np.abs(tens.data) / total
#             sort = np.argsort(norm_data)
#             norm_data = norm_data[sort]

#             cumsum = 0.0
#             for i in range(len(sort)):
#                 cumsum += norm_data[i]
#                 if cumsum > tolerance:
#                     break

#             cutoff = max(i, len(tens.data) - max_bond)

#             errors.append(np.sum(norm_data[:cutoff]))

#             data = tens.data[sort[cutoff:]]
#             all_tensors.append(qtn.Tensor(data, inds=tens.inds, tags=tens.tags))
#             ind_to_sort[tens.inds[0]] = sort[cutoff:]

#         for qt in self.quantum_tensors:
#             qt.ind_tensor = sort_indices(qt.ind_tensor, ind_to_sort)

#         self.qpd_tensors = all_tensors

#         return np.array(errors)

#     def contraction_cost(self) -> int:
#         return self.to_tensor_network().contraction_cost(optimize="auto-hq")

#     def draw(self, color=["QPD", "Q", "wire"], **kwargs):
#         return self.to_tensor_network().draw(color=color, **kwargs)


# def sort_indices(tensor: qtn.Tensor, ind_to_sort: dict[str, np.ndarray]) -> qtn.Tensor:
#     data = tensor.data
#     for ind, sort in ind_to_sort.items():
#         if ind not in tensor.inds:
#             continue
#         dim = tensor.inds.index(ind)
#         data = np.take(data, sort, axis=dim)
#     return qtn.Tensor(data, inds=tensor.inds, tags=tensor.tags)
