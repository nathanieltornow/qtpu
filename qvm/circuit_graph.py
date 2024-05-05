import itertools
from typing import NamedTuple

import networkx as nx
from qiskit.circuit import QuantumCircuit, Qubit
import quimb.tensor as qtn

from qvm.tensor import HybridTensorNetwork, QuantumTensor, InstanceGate
from qvm.virtual_gates import VirtualBinaryGate, VIRTUAL_GATE_GENERATORS, VirtualMove


class CircuitGraphNode(NamedTuple):
    op_id: int
    qubit: Qubit


class CircuitGraph:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        graph = nx.MultiDiGraph()

        qubit_to_node = {}

        for op_id, instr in enumerate(circuit):
            if len(instr.qubits) > 2:
                raise ValueError(
                    f"Only 1 or 2 qubit gates are supported, got {instr.operation}"
                )

            # TODO check if gate is virtualizable

            for qubit in instr.qubits:
                new_node = CircuitGraphNode(op_id=op_id, qubit=qubit)
                graph.add_node(new_node)
                if qubit in qubit_to_node:
                    graph.add_edge(qubit_to_node[qubit], new_node, weight=4)

                qubit_to_node[qubit] = new_node

            if len(instr.qubits) == 2:
                graph.add_edge(
                    CircuitGraphNode(op_id=op_id, qubit=instr.qubits[0]),
                    CircuitGraphNode(op_id=op_id, qubit=instr.qubits[1]),
                    weight=5,
                )

        self._graph = graph

    def hybrid_tn(
        self, connected_components: list[set[CircuitGraphNode]]
    ) -> HybridTensorNetwork:
        quantum_tensors = [
            self._generate_quantum_tensor(component)
            for component in connected_components
        ]
        classical_tensors = self._generate_classical_tensors(connected_components)
        return HybridTensorNetwork(quantum_tensors, classical_tensors)

    def _generate_quantum_tensor(self, node_subset: set[CircuitGraphNode]):
        subcircuit = QuantumCircuit(*self._circuit.qregs, *self._circuit.cregs)

        for op_id, instr in enumerate(self._circuit):
            qubits = instr.qubits

            op_nodes = (
                set(CircuitGraphNode(op_id=op_id, qubit=qubit) for qubit in qubits)
                & node_subset
            )

            for op_node in op_nodes:
                prev_node = self._prev_node_wire(op_node)
                if prev_node is not None and prev_node not in node_subset:
                    edge_id = self._edge_id(prev_node, op_node)
                    subcircuit.append(
                        InstanceGate(
                            num_qubits=1,
                            index="wire_" + edge_id + "_1",
                            instances=VirtualMove().instantiations_qubit1(),
                        ),
                        [op_node.qubit],
                    )

                prev_node = self._prev_node_operation(op_node)
                if prev_node is not None and prev_node not in node_subset:
                    edge_id = self._edge_id(prev_node, op_node)
                    subcircuit.append(
                        InstanceGate(
                            num_qubits=1,
                            index=f"{instr.operation.name}_{edge_id}_1",
                            instances=VIRTUAL_GATE_GENERATORS[instr.operation.name](
                                instr.operation.params
                            ).instantiations_qubit1(),
                        ),
                        [op_node.qubit],
                    )

            if len(op_nodes) == len(qubits):
                subcircuit.append(instr)

            for op_node in op_nodes:
                next_node = self._next_node_wire(op_node)
                if next_node is not None and next_node not in node_subset:
                    edge_id = self._edge_id(op_node, next_node)
                    subcircuit.append(
                        InstanceGate(
                            num_qubits=1,
                            index="wire_" + edge_id + "_0",
                            instances=VirtualMove().instantiations_qubit0(),
                        ),
                        [op_node.qubit],
                    )

                next_node = self._next_node_operation(op_node)
                if next_node is not None and next_node not in node_subset:
                    edge_id = self._edge_id(op_node, next_node)
                    subcircuit.append(
                        InstanceGate(
                            num_qubits=1,
                            index=f"{instr.operation.name}_{edge_id}_0",
                            instances=VIRTUAL_GATE_GENERATORS[instr.operation.name](
                                instr.operation.params
                            ).instantiations_qubit0(),
                        ),
                        [op_node.qubit],
                    )

        return QuantumTensor(subcircuit)

    def _generate_classical_tensors(
        self, connected_components: list[set[CircuitGraphNode]]
    ) -> set[qtn.Tensor]:
        assert set(itertools.chain(*connected_components)) == set(self._graph.nodes)

        tensors = set()

        for op_id, instr in enumerate(self._circuit):
            op, qubits = instr.operation, instr.qubits
            op_nodes = [CircuitGraphNode(op_id=op_id, qubit=qubit) for qubit in qubits]

            for node in op_nodes:
                next_wire_node = self._next_node_wire(node)
                if next_wire_node is not None and not self._nodes_in_same_component(
                    node, next_wire_node, connected_components
                ):
                    edge_id = self._edge_id(node, next_wire_node)
                    tensors.add(
                        qtn.Tensor(
                            VirtualMove().coefficients_2d(),
                            inds=("wire_" + edge_id + "_0", "wire_" + edge_id + "_1"),
                        )
                    )

                next_op_node = self._next_node_operation(node)
                if next_op_node is not None and not self._nodes_in_same_component(
                    node, next_op_node, connected_components
                ):
                    edge_id = self._edge_id(node, next_op_node)
                    tensors.add(
                        qtn.Tensor(
                            VIRTUAL_GATE_GENERATORS[op.name](
                                op.params
                            ).coefficients_2d(),
                            inds=(
                                f"{instr.operation.name}_{edge_id}_0",
                                f"{instr.operation.name}_{edge_id}_1",
                            ),
                        )
                    )
        return tensors

    @staticmethod
    def _nodes_in_same_component(
        node1: CircuitGraphNode,
        node2: CircuitGraphNode,
        connected_components: list[set[CircuitGraphNode]],
    ) -> bool:
        for cc in connected_components:
            if node1 in cc and node2 in cc:
                return True
        return False

    def _next_node_wire(self, node: CircuitGraphNode) -> CircuitGraphNode | None:
        for succ in self._graph.successors(node):
            if succ.qubit == node.qubit:
                return succ
        return None

    def _prev_node_wire(self, node: CircuitGraphNode) -> CircuitGraphNode | None:
        for pred in self._graph.predecessors(node):
            if pred.qubit == node.qubit:
                return pred
        return None

    def _next_node_operation(self, node: CircuitGraphNode) -> CircuitGraphNode | None:
        for succ in self._graph.successors(node):
            if succ.op_id == node.op_id:
                return succ
        return None

    def _prev_node_operation(self, node: CircuitGraphNode) -> CircuitGraphNode | None:
        for pred in self._graph.predecessors(node):
            if pred.op_id == node.op_id:
                return pred
        return None

    def _edge_id(self, u: CircuitGraphNode, v: CircuitGraphNode) -> str:
        return str(hash((u, v)))[:10]
