from networkx.algorithms.community import kernighan_lin_bisection
from qiskit.circuit import Qubit

from qvm.virtual_gates import VIRTUAL_GATE_TYPES

from .dag import DAG, dag_to_qcg
from ._asp import dag_to_asp, qcg_to_asp, get_optimal_symbols


def minimize_qubit_dependencies(dag: DAG, max_virt: int) -> None:
    asp = dag_to_asp(dag)
    asp += _min_dep_asp(max_virt)

    symbols = get_optimal_symbols(asp)

    virtualized_nodes: set[int] = set()
    for symbol in symbols:
        if symbol.name != "vgate":
            continue
        gate_idx = symbol.arguments[0].number
        virtualized_nodes.add(gate_idx)

    for vnode in virtualized_nodes:
        instr = dag.get_node_instr(vnode)
        instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](instr.operation)


def _min_dep_asp(max_virt: int) -> str:
    asp = f"""
    
    num_qubits_of_gate(Gate, N) :- 
        gate(Gate), 
        N = #count{{ Qubit : gate_on_qubit(Gate, Qubit) }}.
    
    {{ vgate(Gate) : gate(Gate), num_qubits_of_gate(Gate, 2) }}.
    
    :- N = #count{{Gate : vgate(Gate)}}, N != {max_virt}.
    
    path(Gate1, Gate2) :- wire(_, Gate1, Gate2), not vgate(Gate1).
    path(Gate1, Gate3) :- path(Gate1, Gate2), path(Gate2, Gate3).
    
    depends_on(Qubit1, Qubit2) :- 
        gate_on_qubit(Gate1, Qubit1),
        gate_on_qubit(Gate2, Qubit2),
        path(Gate1, Gate2),
        Qubit1 != Qubit2.
    
    num_deps(N) :- N = #count{{Qubit1, Qubit2 : depends_on(Qubit1, Qubit2)}}.
    
    
    #minimize{{N : num_deps(N)}}.
    #show vgate/1.
    """
    # for now, we don't use the following constraint
    # num_vgates(N) :- N = #count{{Gate : vgate(Gate)}}.
    # minimize{{N : num_vgates(N)}}.
    return asp
