from qiskit.circuit import QuantumCircuit, Barrier
from qiskit.transpiler import CouplingMap


from qvm.virtual_gates import VirtualBinaryGate, VirtualSWAP
from .gate_virt import (
    cut_gates_bisection,
    cut_gates_optimal,
    minimize_qubit_dependencies,
)
from .dag import DAG
from .qubit_reuse import qubit_reuse
from .wire_cut import cut_wires
from .vqr import apply_virtual_qubit_routing, perfect_virtual_qubit_routing


def cut(
    circuit: QuantumCircuit,
    size_to_reach: int,
    max_overhead: int = 300,
    technique: str = "gate_optimal",
) -> QuantumCircuit:
    dag = DAG(circuit)
    dag.compact()

    if technique == "gate_optimal":
        cut_gates_optimal(dag, size_to_reach)

    elif technique == "gate_bisection":
        cut_gates_bisection(dag, size_to_reach)

    elif technique == "wire_optimal":
        cut_wires(dag, size_to_reach)

    elif technique == "qubit_reuse":
        qubit_reuse(dag, size_to_reach)
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
    dag.compact()
    minimize_qubit_dependencies(dag, max_vgates)
    dag.remove_nodes_of_type(VirtualBinaryGate)
    # qubit_reuse(dag)
    # dag.fragment()
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
        perfect_virtual_qubit_routing(dag, coupling_map, initial_layout)
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
