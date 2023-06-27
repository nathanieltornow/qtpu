from qiskit.circuit import QuantumCircuit, Barrier
from qiskit.transpiler import CouplingMap

from qvm.virtual_gates import VirtualBinaryGate, VirtualSWAP
from .gate_virt import (
    minimize_qubit_dependencies,
)
from .gate_decomposition import decompose_optimal, decompose_qubit_bisection
from .dag import DAG
from .qubit_reuse import random_qubit_reuse
from .wire_cut import cut_wires


def cut(
    circuit: QuantumCircuit,
    size_to_reach: int,
    max_overhead: int = 300,
    technique: str = "gate_optimal",
) -> QuantumCircuit:
    dag = DAG(circuit)

    if technique == "gate_optimal":
        decompose_optimal(dag, size_to_reach)

    elif technique == "gate_bisection":
        decompose_qubit_bisection(dag, size_to_reach)

    elif technique == "wire_optimal":
        cut_wires(dag, size_to_reach)

    elif technique == "qubit_reuse":
        random_qubit_reuse(dag, size_to_reach)
        if len(dag.qubits) > size_to_reach:
            raise ValueError("Qubit reuse did not reach the desired size.")

    elif technique == "gate_qr":
        raise NotImplementedError()

    else:
        raise ValueError(f"Invalid cut-technique {technique}")

    if _overhead(dag) > max_overhead:
        raise ValueError(f"Overhead of {technique} is too high: {_overhead(dag)}")
    dag.fragment()
    return dag.to_circuit()


def virtualize_optimal_gates(
    circuit: QuantumCircuit, max_vgates: int
) -> QuantumCircuit:
    dag = DAG(circuit)
    minimize_qubit_dependencies(dag, max_vgates)
    dag.remove_nodes_of_type(Barrier)
    return dag.to_circuit()


def apply_qubit_reuse(
    circuit: QuantumCircuit, size_to_reach: int = -1
) -> QuantumCircuit:
    dag = DAG(circuit)
    random_qubit_reuse(dag, size_to_reach)
    return dag.to_circuit()


def vqr(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    initial_layout: list[int],
    max_vgates: int = 3,
    technique: str = "perfect",
) -> QuantumCircuit:
    dag = DAG(circuit)
    dag.remove_nodes_of_type(Barrier)

    if technique == "perfect":
        pass
    else:
        raise ValueError(f"Invalid vqr-technique {technique}")

    if _overhead(dag) > 6**max_vgates:
        raise ValueError(f"Overhead of {technique} is too high: {_overhead(dag)}")

    return dag.to_circuit()


def _overhead(dag: DAG) -> int:
    vgates = [
        node
        for node in dag.nodes
        if isinstance(dag.get_node_instr(node).operation, VirtualBinaryGate)
    ]
    wire_cuts = [
        node
        for node in dag.nodes
        if isinstance(dag.get_node_instr(node).operation, VirtualSWAP)
    ]

    if len(vgates) > 0 and len(wire_cuts) > 0:
        raise ValueError("Cannot cut both gates and wires right now")

    oh = 6 ** len(vgates)
    if oh == 1:
        oh = 4 ** len(wire_cuts)
    return oh
