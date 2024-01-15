import networkx as nx
from networkx.classes.graph import Graph
from qiskit.circuit import QuantumCircuit, Barrier

from qvm.instructions import VirtualBinaryGate


class TNGraph(nx.Graph):
    def __init__(self, circuit: QuantumCircuit, wire_cost: int = 4, gate_cost: int = 5):
        self._circuit = circuit

        super().__init__()
        # qubit_idx -> node_idx
        current_nodes: dict[int, int] = {i: -1 for i in range(len(circuit.qubits))}

        nodeidx = 0
        for instr_idx, instr in enumerate(circuit):
            added_nodes: list[int] = []

            if len(instr.qubits) == 1:
                continue

            for qubit in instr.qubits:
                qubit_idx = circuit.qubits.index(qubit)

                self.add_node(nodeidx, qubit_idx=qubit_idx, instr_idx=instr_idx)
                added_nodes.append(nodeidx)

                if current_nodes[qubit_idx] != -1:
                    self.add_edge(current_nodes[qubit_idx], nodeidx, weight=wire_cost)
                current_nodes[qubit_idx] = nodeidx

                nodeidx += 1

            if isinstance(instr.operation, VirtualBinaryGate | Barrier):
                continue

            for i in range(len(added_nodes) - 1):
                self.add_edge(added_nodes[i], added_nodes[i + 1], weight=gate_cost)

    def copy(self, as_view=False) -> Graph:
        return TNGraph(self._circuit)

    def is_wire_edge(self, u: int, v: int) -> bool:
        return (
            self.has_edge(u, v)
            and self.nodes[u]["qubit_idx"] == self.nodes[v]["qubit_idx"]
        )

    def is_gate_edge(self, u: int, v: int) -> bool:
        return (
            self.has_edge(u, v)
            and self.nodes[u]["instr_idx"] == self.nodes[v]["instr_idx"]
        )


class QubitGraph(nx.Graph):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__()
        self.add_nodes_from(range(len(circuit.qubits)))

        for instr in circuit:
            if (
                len(instr.qubits) == 1
                or isinstance(instr.operation, Barrier)
                or isinstance(instr.operation, VirtualBinaryGate)
            ):
                continue
            for qubit1, qubit2 in zip(instr.qubits, instr.qubits[1:]):
                q1_idx, q2_idx = (
                    circuit.qubits.index(qubit1),
                    circuit.qubits.index(qubit2),
                )
                if self.has_edge(q1_idx, q2_idx):
                    self[q1_idx][q2_idx]["weight"] += 1
                self.add_edge(
                    circuit.qubits.index(qubit1), circuit.qubits.index(qubit2), weight=1
                )


class PortGraph(nx.Graph):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__()
        current_nodes: dict[int, tuple[int, int]] = {
            qubit: (-1, -1) for qubit in circuit.qubits
        }

        for instr_idx, instr in enumerate(circuit):
            self.add_node(instr_idx)

            for qubit_idx, qubit in enumerate(instr.qubits):
                if current_nodes[qubit] != (-1, -1):
                    prev_node, prev_qubit_idx = current_nodes[qubit]
                    self.add_edge(
                        prev_node,
                        instr_idx,
                        from_qubit=prev_qubit_idx,
                        to_qubit=qubit_idx,
                    )

                current_nodes[qubit] = (instr_idx, qubit_idx)
