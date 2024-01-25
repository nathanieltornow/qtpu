import networkx as nx
from qiskit.circuit import QuantumCircuit, Barrier, QuantumRegister, Qubit

from qvm.virtual_circuit import VirtualCircuit
from qvm.instructions import VirtualBinaryGate, WireCut
from qvm.virtual_gates import VIRTUAL_GATE_GENERATORS, VirtualMove

from ._estimator import SuccessEstimator
from .contraction_tree import ContractionTree, Bisector


def cut(
    circuit: QuantumCircuit,
    success_estimator: SuccessEstimator,
    bisector: Bisector | None = None,
    max_contraction_cost: int = 1000,
    alpha: float = 0.5,
    max_iters: int = 100,
) -> VirtualCircuit:
    if any(isinstance(instr.operation, VirtualBinaryGate) for instr in circuit):
        raise ValueError("Circuit already contains virtual gates")

    if bisector is None:
        bisector = Bisector()

    circuit_graph = CircuitGraph(circuit)
    contraction_tree = ContractionTree(circuit_graph.get_nx_graph(), bisector)

    current_score = (1 - alpha) * success_estimator.estimate(circuit)

    for _ in range(max_iters):
        contraction_tree.bisect()
        new_circuit = circuit_graph.generate_circuit(contraction_tree.removed_edges())

        success_est = success_estimator.estimate(new_circuit)
        if success_est <= 0.0:
            continue

        knit_cost = contraction_tree.contraction_cost() / max_contraction_cost
        if knit_cost > 1.0:
            raise ValueError("Contraction cost too high")

        score = alpha * (1 - knit_cost) + (1 - alpha) * success_est

        if score > current_score:
            current_score = score
            circuit = new_circuit
        else:
            break

    return circuit


