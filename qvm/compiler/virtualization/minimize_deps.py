from qiskit.circuit import QuantumCircuit

from qvm.compiler._asp import get_optimal_symbols, dag_to_asp
from qvm.compiler._types import CutCompiler
from qvm.compiler.dag import DAG
from qvm.virtual_gates import VIRTUAL_GATE_TYPES


class MinimizeQubitDependencies(CutCompiler):
    def __init__(self, max_vgates: int) -> None:
        self._max_vgates = max_vgates
        super().__init__()

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        dag = DAG(circuit)
        self._run_on_dag(dag)
        return dag.to_circuit()

    def _run_on_dag(self, dag: DAG) -> None:
        asp = dag_to_asp(dag)
        asp += self._min_dep_asp()

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

    def _min_dep_asp(self) -> str:
        asp = f"""
        {{ vgate(Gate) : gate(Gate) }}.
        
        :- N = #count{{Gate : vgate(Gate)}}, N > {self._max_vgates}.
        
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
        #minimize{{N : num_vgates(N)}}.
        #show vgate/1.
        """
        return asp
