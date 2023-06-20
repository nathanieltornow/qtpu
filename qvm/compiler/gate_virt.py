from qvm.virtual_gates import VIRTUAL_GATE_TYPES
from qvm.dag import DAG

from ._asp import dag_to_asp


def minimize_qubit_dependencies(dag: DAG, max_virt: int) -> None:
    from clingo.control import Control

    asp = dag_to_asp(dag)
    asp += _min_dep_asp(max_virt)

    control = Control()
    control.configuration.solve.models = 0  # type: ignore
    control.add("base", [], asp)
    control.ground([("base", [])])
    solve_result = control.solve(yield_=True)  # type: ignore
    opt_model = None
    for model in solve_result:  # type: ignore
        opt_model = model

    if opt_model is None:
        raise ValueError("No solution found.")

    virtualized_nodes: set[int] = set()
    for symbol in opt_model.symbols(shown=True):
        if symbol.name != "vgate":
            continue
        gate_idx = symbol.arguments[0].number
        virtualized_nodes.add(gate_idx)

    for vnode in virtualized_nodes:
        instr = dag.get_node_instr(vnode)
        instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](instr.operation)


def _min_dep_asp(max_virt: int) -> str:
    asp = f"""
    {{ vgate(Gate) : gate(Gate, _, _) }}.
    
    :- N = #count{{Gate : vgate(Gate)}}, N > {max_virt}.
    
    gate_on_qubit(Gate, Qubit) :- gate(Gate, Qubit).
    gate_on_qubit(Gate, Qubit) :- gate(Gate, Qubit, _).
    gate_on_qubit(Gate, Qubit) :- gate(Gate, _, Qubit).
    
    path(Gate1, Gate2) :- wire(_, Gate1, Gate2), not vgate(Gate1).
    path(Gate1, Gate3) :- path(Gate1, Gate2), path(Gate2, Gate3).
    
    depends_on(Qubit1, Qubit2) :- 
        gate_on_qubit(Gate1, Qubit1),
        gate_on_qubit(Gate2, Qubit2),
        path(Gate1, Gate2),
        Qubit1 != Qubit2.
    
    num_deps(N) :- N = #count{{Qubit1, Qubit2 : depends_on(Qubit1, Qubit2)}}.
    
    num_vgates(N) :- N = #count{{Gate : vgate(Gate)}}.
    
    #minimize{{N : num_deps(N)}}.
    % # minimize{{N : num_vgates(N)}}.
    #show vgate/1.
    """
    return asp
