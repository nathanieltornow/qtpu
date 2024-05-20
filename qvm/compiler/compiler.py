from typing import Any

from qiskit.circuit import QuantumCircuit

from qvm.graph import CircuitGraph
from qvm.tensor import HybridTensorNetwork
from qvm.compiler.optimizer import Optimizer


def compile_circuit(
    circuit: QuantumCircuit,
    optimizer: Optimizer,
) -> HybridTensorNetwork:
    # frontend
    cg = CircuitGraph(circuit)

    # optimizer
    ccs = optimizer.optimize(cg)
    hybrid_tn = cg.hybrid_tn(connected_components=ccs)

    return hybrid_tn
