import abc
import itertools
from dataclasses import dataclass

import cotengra as ctg
from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit

from qvm.instructions import InstanceGate
from qvm.virtual_gates import VirtualMove, VIRTUAL_GATE_GENERATORS
from qvm.tensor import QuantumTensor, ClassicalTensor, HybridTensorNetwork


@dataclass(frozen=True)
class NodeInfo:
    op_idx: int  # Index of the operation in the circuit
    abs_qubit: Qubit  # Absolute qubit in the circuit
    rel_qubit: int  # Relative qubit in the operation


class HybridCircuitIface(abc.ABC):
    @abc.abstractmethod
    def inputs(self) -> list[tuple[str, ...]]:
        """The inputs of the TN-like representation of the circuit.

        Returns:
            list[tuple[str, ...]]: The inputs of the TN-like representation of the circuit.
        """
        ...

    def output(self) -> tuple[str, ...]:
        """The output of the TN-like representation of the circuit.

        Returns:
            tuple[str, ...]: The output of the TN-like representation of the circuit.
        """
        return tuple()

    @abc.abstractmethod
    def size_dict(self) -> dict[str, int]:
        """The size dictionary of each index the TN-like representation of the circuit.

        Returns:
            dict[str, int]: The size dictionary of the TN-like representation of the circuit.
        """
        ...

    @abc.abstractmethod
    def node_infos(self) -> dict[int, set[NodeInfo]]:
        """Node information for each node in the TN-like representation of the circuit.

        Returns:
            dict[int, set[NodeInfo]]: The node information for each node.
        """
        ...

    @abc.abstractmethod
    def quantum_tensor(self, node_subset: set[int]) -> QuantumTensor:
        """Generates a quantum tensor from a subset of nodes.

        Args:
            node_subset (set[int]): A subset of nodes of the TN-like representation of the circuit.

        Returns:
            QuantumTensor: The quantum tensor generated from the subset of nodes.
        """
        ...

    @abc.abstractmethod
    def hybrid_tn(self, node_subsets: list[set[int]]) -> HybridTensorNetwork:
        """Generates a hybrid tensor network from a list of node subsets.

        Args:
            node_subsets (list[set[int]]): The list of node subsets.

        Returns:
            HybridTensorNetwork: The hybrid tensor network generated from the list of node subsets.
        """
        ...

    def num_qubits(self, node_subset: set[int]) -> int:
        """Computes the number of qubits in a subset of nodes.

        Args:
            node_subset (set[int]): The node subset corresponding to a quantum tensor.

        Returns:
            int: The number of qubits.
        """
        return len(
            set(
                itertools.chain.from_iterable(
                    set(info.abs_qubit for info in self.node_infos()[node])
                    for node in node_subset
                )
            )
        )


class HybridCircuitIR(HybridCircuitIface):
    def __init__(self, circuit: QuantumCircuit) -> None:
        node_infos: dict[int, NodeInfo] = {}
        op_nodes: list[set[int]] = []
        inputs: list[tuple[str, ...]] = []
        size_dict: dict[str, int] = {}

        edge_index = 200
        current_nodes: dict[Qubit, int] = {}
        for op_id, instr in enumerate(circuit):
            if len(instr.qubits) > 2:
                raise ValueError(
                    f"Only 1 or 2 qubit gates are supported, got {instr.operation}"
                )

            op_nodes.append(set())

            for i, qubit in enumerate(instr.qubits):
                inputs.append(tuple())
                node_id = len(inputs) - 1
                op_nodes[-1].add(node_id)
                node_infos[node_id] = NodeInfo(
                    op_idx=op_id, abs_qubit=qubit, rel_qubit=i
                )

                if qubit in current_nodes:
                    inputs[current_nodes[qubit]] += (str(chr(edge_index)),)
                    inputs[node_id] += (str(chr(edge_index)),)
                    size_dict[str(chr(edge_index))] = 4

                    edge_index += 1

                current_nodes[qubit] = node_id

            if len(instr.qubits) == 2:
                assert instr.operation.name in VIRTUAL_GATE_GENERATORS

                inputs[-2] += (str(chr(edge_index)),)
                inputs[-1] += (str(chr(edge_index)),)
                size_dict[str(chr(edge_index))] = 5

                edge_index += 1

        self._circuit = circuit
        self._op_nodes = op_nodes
        self._node_infos = node_infos
        self._hypergraph = ctg.HyperGraph(inputs, tuple(), size_dict)

    def node_infos(self) -> dict[int, NodeInfo]:
        return {node: {info} for node, info in self._node_infos.items()}

    def inputs(self) -> list[tuple[str, ...]]:
        return self._hypergraph.inputs.copy()

    def size_dict(self) -> dict[str, int]:
        return self._hypergraph.size_dict.copy()

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit.copy()

    @property
    def op_nodes(self) -> list[set[int]]:
        return self._op_nodes.copy()

    @property
    def hypergraph(self) -> ctg.HyperGraph:
        return self._hypergraph.copy()

    def hybrid_tn(self, node_subsets: list[set[int]]) -> HybridTensorNetwork:
        assert set(itertools.chain(*node_subsets)) == set(
            range(self._hypergraph.num_nodes)
        )

        quantum_tensors = []
        removed_edges = set()
        for node_subset in node_subsets:
            qt, re = self.quantum_tensor(set(node_subset), True)
            quantum_tensors.append(qt)
            removed_edges |= re

        classical_tensors = [self._classical_tensor(u, v) for u, v in removed_edges]

        return HybridTensorNetwork(quantum_tensors, classical_tensors)

    def _prev_nodes(self, node: int) -> list[int]:
        return sorted(
            neigh for neigh in self._hypergraph.neighbors(node) if neigh < node
        )

    def _next_nodes(self, node: int) -> list[int]:
        return sorted(
            neigh for neigh in self._hypergraph.neighbors(node) if neigh > node
        )

    def _instance_gate(self, this: int, other: int) -> InstanceGate:
        assert this != other

        if self._node_infos[this].op_idx == self._node_infos[other].op_idx:
            op = self._circuit[self._node_infos[this].op_idx].operation
            vgate = VIRTUAL_GATE_GENERATORS[op.name](op.params)
        elif self._node_infos[this].abs_qubit == self._node_infos[other].abs_qubit:
            vgate = VirtualMove()
        else:
            raise ValueError("Invalid edge")

        if this < other:
            instances = vgate.instances_q0()
        else:
            instances = vgate.instances_q1()

        s = f"{this}_{other}" if this < other else f"{other}_{this}"

        return InstanceGate(
            num_qubits=1,
            index=f"{s}_{0 if this < other else 1}",
            instances=instances,
        )

    def quantum_tensor(
        self, node_subset: set[int], return_edges: bool = False
    ) -> QuantumTensor | tuple[QuantumTensor, set[tuple[int, int]]]:
        sub_circuit = QuantumCircuit(*self._circuit.qregs, *self._circuit.cregs)

        removed_edges = set()

        for instr, node_ids in zip(self._circuit, self._op_nodes):

            relevant_nodes = set(node_ids) & node_subset
            if len(relevant_nodes) == 0:
                continue

            for node in relevant_nodes:
                for prev_node in self._prev_nodes(node):
                    if prev_node not in node_subset:
                        sub_circuit.append(
                            self._instance_gate(node, prev_node),
                            [self._node_infos[node].abs_qubit],
                        )
                        removed_edges.add((prev_node, node))

            if len(relevant_nodes) == len(instr.qubits):
                sub_circuit.append(instr)

            for node in relevant_nodes:
                for next_node in self._next_nodes(node):
                    if next_node not in node_subset:
                        sub_circuit.append(
                            self._instance_gate(node, next_node),
                            [self._node_infos[node].abs_qubit],
                        )
                        removed_edges.add((node, next_node))

        sub_circuit = remove_idle_qubits(sub_circuit)

        if return_edges:
            return QuantumTensor(sub_circuit), removed_edges

        return QuantumTensor(sub_circuit)

    def _classical_tensor(self, u: int, v: int) -> ClassicalTensor:
        u_info, v_info = self._node_infos[u], self._node_infos[v]
        if u_info.abs_qubit == v_info.abs_qubit:
            return ClassicalTensor(
                VirtualMove().coefficients_2d(),
                inds=(f"{u}_{v}_0", f"{u}_{v}_1"),
            )

        assert u_info.op_idx == v_info.op_idx

        op = self._circuit[u_info.op_idx].operation
        return ClassicalTensor(
            VIRTUAL_GATE_GENERATORS[op.name](op.params).coefficients_2d(),
            inds=(f"{u}_{v}_0", f"{u}_{v}_1"),
        )


def remove_idle_qubits(circuit: QuantumCircuit) -> QuantumCircuit:
    used_qubits = sorted(
        set(itertools.chain(*[instr.qubits for instr in circuit])),
        key=lambda q: circuit.qubits.index(q),
    )
    qreg = QuantumRegister(len(used_qubits), "q")
    qubit_map = {
        old_qubit: new_qubit for old_qubit, new_qubit in zip(used_qubits, qreg)
    }
    new_circuit = QuantumCircuit(qreg, *circuit.cregs)

    for instr in circuit:
        new_qubits = [qubit_map[qubit] for qubit in instr.qubits]
        new_circuit.append(instr.operation, new_qubits, instr.clbits)

    return new_circuit
