from dataclasses import dataclass

import cotengra as ctg
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit_addon_cutting.qpd import QPDBasis
from qiskit_addon_cutting.instructions import Move

from qtpu.transforms import insert_cuts


@dataclass(frozen=True)
class NodeInfo:
    op_idx: int  # Index of the operation in the circuit
    abs_qubit: Qubit  # Absolute qubit in the circuit
    rel_qubit: int  # Relative qubit in the operation


class HybridCircuitIR:
    def __init__(self, circuit: QuantumCircuit) -> None:
        node_infos: list[NodeInfo] = []
        op_nodes: list[set[int]] = []
        inputs: list[tuple[str, ...]] = []
        size_dict: dict[str, int] = {}

        edge_index = 200
        current_nodes: dict[Qubit, int] = {}
        for op_id, instr in enumerate(circuit):

            op_nodes.append(set())

            for i, qubit in enumerate(instr.qubits):
                inputs.append(tuple())
                op_nodes[-1].add(len(inputs) - 1)
                node_infos.append(NodeInfo(op_idx=op_id, abs_qubit=qubit, rel_qubit=i))

                if qubit in current_nodes:
                    inputs[current_nodes[qubit]] += (str(chr(edge_index)),)
                    inputs[-1] += (str(chr(edge_index)),)
                    size_dict[str(chr(edge_index))] = round(
                        QPDBasis.from_instruction(Move()).overhead
                    )
                    # size_dict[str(chr(edge_index))] = 16

                    edge_index += 1

                current_nodes[qubit] = len(inputs) - 1

            try:
                # edge_weigth = round(QPDBasis.from_instruction(instr.operation).overhead)
                edge_weigth = QPDBasis.from_instruction(instr.operation).overhead

                # edge_weigth = len(QPDBasis.from_instruction(instr.operation).coeffs)
            except ValueError:
                edge_weigth = 1e15

            for i in range(len(instr.qubits) - 1):
                inputs[-i - 2] += (str(chr(edge_index)),)
                inputs[-i - 1] += (str(chr(edge_index)),)
                size_dict[str(chr(edge_index))] = edge_weigth

                edge_index += 1

        self._circuit = circuit
        self._op_nodes = op_nodes
        self._node_infos = node_infos
        self._hypergraph = ctg.HyperGraph(inputs, tuple(), size_dict)

    def node_info(self, node: int) -> NodeInfo:
        return self._node_infos[node]

    def node_infos(self) -> list[set[NodeInfo]]:
        return [{info} for info in self._node_infos]

    def contraction_tree(self) -> ctg.ContractionTree:
        return ctg.ContractionTree(
            self._hypergraph.inputs,
            tuple(),
            self._hypergraph.size_dict,
            track_flops=True,
            track_childless=True,
        )

    def num_qubits(self, node_subset: set[int]) -> int:
        return len(set(self._node_infos[node].abs_qubit for node in node_subset))

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit.copy()

    @property
    def op_nodes(self) -> list[set[int]]:
        return self._op_nodes.copy()

    @property
    def hypergraph(self) -> ctg.HyperGraph:
        return self._hypergraph.copy()

    def cut_circuit(self, node_subsets: list[set[int]]) -> QuantumCircuit:
        node_to_subset = {
            node: i for i, subset in enumerate(node_subsets) for node in subset
        }
        cut_edges = [
            (u, v)
            for u, v in self._hypergraph.edges.values()
            if node_to_subset[u] != node_to_subset[v]
        ]
        wire_cuts = {
            (self._node_infos[u].op_idx, self._node_infos[u].rel_qubit)
            for u, v in cut_edges
            if self._node_infos[u].abs_qubit == self._node_infos[v].abs_qubit
        }
        gate_cuts = {
            self._node_infos[u].op_idx
            for u, v in cut_edges
            if self._node_infos[u].op_idx == self._node_infos[v].op_idx
        }
        return insert_cuts(self._circuit, gate_cuts, wire_cuts)

    # def hybrid_tn(self, node_subsets: list[set[int]]) -> HybridTensorNetwork:
    #     assert set(itertools.chain(*node_subsets)) == set(
    #         range(self._hypergraph.num_nodes)
    #     )

    #     quantum_tensors = []
    #     removed_edges = set()
    #     for node_subset in node_subsets:
    #         qt, re = self.quantum_tensor(set(node_subset), True)
    #         quantum_tensors.append(qt)
    #         removed_edges |= re

    #     classical_tensors = [self._classical_tensor(u, v) for u, v in removed_edges]

    #     return HybridTensorNetwork(quantum_tensors, classical_tensors)

    # def _prev_nodes(self, node: int) -> list[int]:
    #     return sorted(
    #         neigh for neigh in self._hypergraph.neighbors(node) if neigh < node
    #     )

    # def _next_nodes(self, node: int) -> list[int]:
    #     return sorted(
    #         neigh for neigh in self._hypergraph.neighbors(node) if neigh > node
    #     )

    # def _instance_gate(self, this: int, other: int) -> InstanceGate:
    #     assert this != other

    #     if self._node_infos[this].op_idx == self._node_infos[other].op_idx:
    #         op = self._circuit[self._node_infos[this].op_idx].operation
    #         vgate = VIRTUAL_GATE_GENERATORS[op.name](op.params)
    #     elif self._node_infos[this].abs_qubit == self._node_infos[other].abs_qubit:
    #         vgate = VirtualMove()
    #     else:
    #         raise ValueError("Invalid edge")

    #     if this < other:
    #         instances = vgate.instances_q0()
    #     else:
    #         instances = vgate.instances_q1()

    #     s = f"{this}_{other}" if this < other else f"{other}_{this}"

    #     return InstanceGate(
    #         num_qubits=1,
    #         index=f"{s}_{0 if this < other else 1}",
    #         instances=instances,
    #     )

    # def quantum_tensor(
    #     self, node_subset: set[int], return_edges: bool = False
    # ) -> QuantumTensor | tuple[QuantumTensor, set[tuple[int, int]]]:
    #     sub_circuit = QuantumCircuit(*self._circuit.qregs, *self._circuit.cregs)

    #     removed_edges = set()

    #     for instr, node_ids in zip(self._circuit, self._op_nodes):

    #         relevant_nodes = set(node_ids) & node_subset
    #         if len(relevant_nodes) == 0:
    #             continue

    #         for node in relevant_nodes:
    #             for prev_node in self._prev_nodes(node):
    #                 if prev_node not in node_subset:
    #                     sub_circuit.append(
    #                         self._instance_gate(node, prev_node),
    #                         [self._node_infos[node].abs_qubit],
    #                     )
    #                     removed_edges.add((prev_node, node))

    #         if len(relevant_nodes) == len(instr.qubits):
    #             sub_circuit.append(instr)

    #         for node in relevant_nodes:
    #             for next_node in self._next_nodes(node):
    #                 if next_node not in node_subset:
    #                     sub_circuit.append(
    #                         self._instance_gate(node, next_node),
    #                         [self._node_infos[node].abs_qubit],
    #                     )
    #                     removed_edges.add((node, next_node))

    #     sub_circuit = remove_idle_qubits(sub_circuit)

    #     if return_edges:
    #         return QuantumTensor(sub_circuit), removed_edges

    #     return QuantumTensor(sub_circuit)

    # def _classical_tensor(self, u: int, v: int) -> ClassicalTensor:
    #     u_info, v_info = self._node_infos[u], self._node_infos[v]
    #     if u_info.abs_qubit == v_info.abs_qubit:
    #         return ClassicalTensor(
    #             VirtualMove().coefficients_2d(),
    #             inds=(f"{u}_{v}_0", f"{u}_{v}_1"),
    #         )

    #     assert u_info.op_idx == v_info.op_idx

    #     op = self._circuit[u_info.op_idx].operation
    #     return ClassicalTensor(
    #         VIRTUAL_GATE_GENERATORS[op.name](op.params).coefficients_2d(),
    #         inds=(f"{u}_{v}_0", f"{u}_{v}_1"),
    #     )


# def remove_idle_qubits(circuit: QuantumCircuit) -> QuantumCircuit:
#     used_qubits = sorted(
#         set(itertools.chain(*[instr.qubits for instr in circuit])),
#         key=lambda q: circuit.qubits.index(q),
#     )
#     qreg = QuantumRegister(len(used_qubits), "q")
#     qubit_map = {
#         old_qubit: new_qubit for old_qubit, new_qubit in zip(used_qubits, qreg)
#     }
#     new_circuit = QuantumCircuit(qreg, *circuit.cregs)

#     for instr in circuit:
#         new_qubits = [qubit_map[qubit] for qubit in instr.qubits]
#         new_circuit.append(instr.operation, new_qubits, instr.clbits)

#     return new_circuit


# class IRInterface(abc.ABC):
#     @abc.abstractmethod
#     def contraction_tree(self) -> ctg.ContractionTree: ...

#     @abc.abstractmethod
#     def quantum_tensor(self, node_subset: set[int]) -> QuantumTensor: ...

#     @abc.abstractmethod
#     def node_infos(self, node: int) -> set[NodeInfo]: ...

#     @abc.abstractmethod
#     def hybrid_tn(self, node_subsets: list[set[int]]) -> HybridTensorNetwork: ...

#     def num_qubits(self, node_subset: set[int]) -> int:
#         return len(
#             set(
#                 itertools.chain.from_iterable(
#                     set(info.abs_qubit for info in self.node_infos(node))
#                     for node in node_subset
#                 )
#             )
#         )
