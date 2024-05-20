import itertools
from typing import NamedTuple

import networkx as nx
from qiskit.circuit import QuantumCircuit, Qubit, QuantumRegister
import quimb.tensor as qtn

from qvm.tensor import HybridTensorNetwork, QuantumTensor, InstanceGate
from qvm.virtual_gates import VIRTUAL_GATE_GENERATORS, VirtualMove


class CircuitGraphNode(NamedTuple):
    op_id: int
    qubit: Qubit


class CircuitGraph:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        graph = nx.DiGraph()

        qubit_to_node = {}

        for op_id, instr in enumerate(circuit):
            if len(instr.qubits) > 2:
                raise ValueError(
                    f"Only 1 or 2 qubit gates are supported, got {instr.operation}"
                )


            for qubit in instr.qubits:
                new_node = CircuitGraphNode(op_id=op_id, qubit=qubit)
                graph.add_node(new_node)
                if qubit in qubit_to_node:
                    graph.add_edge(qubit_to_node[qubit], new_node, weight=4)

                qubit_to_node[qubit] = new_node

            if len(instr.qubits) == 2:
                assert instr.operation.name in VIRTUAL_GATE_GENERATORS
                graph.add_edge(
                    CircuitGraphNode(op_id=op_id, qubit=instr.qubits[0]),
                    CircuitGraphNode(op_id=op_id, qubit=instr.qubits[1]),
                    weight=5,
                )

        self._graph = graph

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit.copy()

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    def hybrid_tn(
        self, connected_components: list[set[CircuitGraphNode]]
    ) -> HybridTensorNetwork:
        quantum_tensors = [
            self.generate_quantum_tensor(component)
            for component in connected_components
        ]
        classical_tensors = self.generate_classical_tensors(connected_components)
        return HybridTensorNetwork(quantum_tensors, classical_tensors)

    def generate_quantum_tensor(self, node_subset: set[CircuitGraphNode]):
        subcircuit = QuantumCircuit(*self._circuit.qregs, *self._circuit.cregs)

        for op_id, instr in enumerate(self._circuit):
            qubits = instr.qubits

            op_nodes = (
                set(CircuitGraphNode(op_id=op_id, qubit=qubit) for qubit in qubits)
                & node_subset
            )

            for op_node in op_nodes:
                prev_node = self.prev_node_wire(op_node)
                if prev_node is not None and prev_node not in node_subset:
                    edge_id = self.edge_id(prev_node, op_node)

                    subcircuit.append(
                        InstanceGate(
                            num_qubits=1,
                            index=edge_id + "_1",
                            instances=VirtualMove().instances_q1(),
                        ),
                        [op_node.qubit],
                    )

                prev_node = self.prev_node_operation(op_node)
                if prev_node is not None and prev_node not in node_subset:
                    edge_id = self.edge_id(prev_node, op_node)
                    subcircuit.append(
                        InstanceGate(
                            num_qubits=1,
                            index=f"{edge_id}_1",
                            instances=VIRTUAL_GATE_GENERATORS[instr.operation.name](
                                instr.operation.params
                            ).instances_q1(),
                        ),
                        [op_node.qubit],
                    )

            if len(op_nodes) == len(qubits):
                subcircuit.append(instr)

            for op_node in op_nodes:
                next_node = self.next_node_wire(op_node)
                if next_node is not None and next_node not in node_subset:
                    edge_id = self.edge_id(op_node, next_node)
                    subcircuit.append(
                        InstanceGate(
                            num_qubits=1,
                            index=edge_id + "_0",
                            instances=VirtualMove().instances_q0(),
                        ),
                        [op_node.qubit],
                    )

                next_node = self.next_node_operation(op_node)
                if next_node is not None and next_node not in node_subset:
                    edge_id = self.edge_id(op_node, next_node)
                    subcircuit.append(
                        InstanceGate(
                            num_qubits=1,
                            index=f"{edge_id}_0",
                            instances=VIRTUAL_GATE_GENERATORS[instr.operation.name](
                                instr.operation.params
                            ).instances_q0(),
                        ),
                        [op_node.qubit],
                    )
        return QuantumTensor(remove_idle_qubits(subcircuit))

    def edge_to_classical_tensor(
        self, u: CircuitGraphNode, v: CircuitGraphNode
    ) -> qtn.Tensor:
        assert self._graph.has_edge(u, v)

        edge_id = self.edge_id(u, v)
        if u.qubit == v.qubit:
            return qtn.Tensor(
                VirtualMove().coefficients_2d(),
                inds=(edge_id + "_0", edge_id + "_1"),
            )
        elif u.op_id == v.op_id:
            return qtn.Tensor(
                VIRTUAL_GATE_GENERATORS[self._circuit[u.op_id].operation.name](
                    self._circuit[u.op_id].operation.params
                ).coefficients_2d(),
                inds=(edge_id + "_0", edge_id + "_1"),
            )

    def generate_classical_tensors(
        self, connected_components: list[set[CircuitGraphNode]]
    ) -> list[qtn.Tensor]:
        assert set(itertools.chain(*connected_components)) == set(self._graph.nodes)

        tensors = []

        for op_id, instr in enumerate(self._circuit):
            op, qubits = instr.operation, instr.qubits
            op_nodes = [CircuitGraphNode(op_id=op_id, qubit=qubit) for qubit in qubits]

            for node in op_nodes:
                next_wire_node = self.next_node_wire(node)
                if next_wire_node is not None and not self._nodes_in_same_component(
                    node, next_wire_node, connected_components
                ):
                    tensors.append(self.edge_to_classical_tensor(node, next_wire_node))

                next_op_node = self.next_node_operation(node)
                if next_op_node is not None and not self._nodes_in_same_component(
                    node, next_op_node, connected_components
                ):

                    tensors.append(self.edge_to_classical_tensor(node, next_op_node))

        return tensors

    def next_node_wire(self, node: CircuitGraphNode) -> CircuitGraphNode | None:
        for succ in self._graph.successors(node):
            if succ.qubit == node.qubit:
                return succ
        return None

    def prev_node_wire(self, node: CircuitGraphNode) -> CircuitGraphNode | None:
        for pred in self._graph.predecessors(node):
            if pred.qubit == node.qubit:
                return pred
        return None

    def next_node_operation(self, node: CircuitGraphNode) -> CircuitGraphNode | None:
        for succ in self._graph.successors(node):
            if succ.op_id == node.op_id:
                return succ
        return None

    def prev_node_operation(self, node: CircuitGraphNode) -> CircuitGraphNode | None:
        for pred in self._graph.predecessors(node):
            if pred.op_id == node.op_id:
                return pred
        return None

    def edge_id(self, u: CircuitGraphNode, v: CircuitGraphNode) -> str:
        assert self._graph.has_edge(u, v)
        return str(hash((u, v)))[:10]

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
