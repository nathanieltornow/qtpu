import networkx as nx

from qiskit.circuit import QuantumCircuit, Qubit, Instruction


class CircuitGraph:
    def __init__(self) -> None:
        pass

    @property
    def graph(self) -> nx.Graph:
        pass


class CircuitIRNode:
    def __init__(self, op_idx: int, operation: Instruction, qubits: tuple[Qubit, ...]):
        self.op_idx = op_idx
        self.operation = operation
        self.qubits = qubits

    def __hash__(self) -> int:
        return hash((self.op_idx, self.qubits))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircuitIRNode):
            return False
        return self.op_idx == other.op_idx and self.qubits == other.qubits



@dataclass
class CircuitTensor:
    circuit: QuantumCircuit
    virtual_gates: list[VirtualBinaryGate]
    param_tensor: np.ndarray


class CircuitIR:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._graph = nx.Graph()

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def nodes(self) -> list[CircuitIRNode]:
        pass

    def edges(self) -> list[tuple[CircuitIRNode, CircuitIRNode]]:
        pass

    def operation_edges(self) -> set[tuple[CircuitIRNode, CircuitIRNode]]:
        pass

    def wire_edges(self) -> set[tuple[CircuitIRNode, CircuitIRNode]]:
        pass

    def generate_subcircuit(self, node_subset: set[CircuitIRNode]) -> CircuitTensor:
        # 1. Check which edges are dangling

        pass


def subcircuit_to_tensor(
    circuit_graph: CircuitGraph, nodes: CircuitIRNode
) -> CircuitTensor:
    pass


class KnitTree:
    def __init__(self, cg: CircuitGraph) -> None:
        pass

    @property
    def left(self) -> "KnitTree" | None:
        pass

    @property
    def right(self) -> "KnitTree" | None:
        pass

    def edges_between(self) -> set[tuple[int, int]]:
        pass

    def knit_cost(self) -> int:
        pass

    def divide(self, left_nodes: set[int]) -> None:
        pass


class SubCircuitGraph:

    def __init__(self, circuit_graph: CircuitGraph, nodes: set[int]) -> None:
        pass

    def circuit(self) -> QuantumCircuit:
        pass

    def params(self) -> dict[str, Any]:
        pass
