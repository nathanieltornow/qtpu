import abc
from collections import Counter

import networkx as nx
from qiskit.circuit import QuantumCircuit, Qubit, Barrier

from qvm.compiler.asp import get_optimal_symbols, dag_to_asp
from qvm.compiler.types import CutCompiler
from qvm.compiler.dag import DAG


class ReduceQubitDependenciesCompiler(CutCompiler, abc.ABC):
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        dag = DAG(circuit)
        dag.compact()
        self._run_on_dag(dag)
        dag.fragment()
        return dag.to_circuit()

    @abc.abstractmethod
    def _run_on_dag(self, dag: DAG) -> None:
        ...


class CircularDependencyBreaker(ReduceQubitDependenciesCompiler):
    def __init__(self, max_vgates: int) -> None:
        self._max_vgates = max_vgates
        super().__init__()

    def _run_on_dag(self, dag: DAG) -> None:
        qubit_depends_on: dict[Qubit, Counter[Qubit]] = {
            qubit: Counter() for qubit in dag.qubits
        }
        budget = self._max_vgates
        nodes = nx.topological_sort(dag)
        for node in nodes:
            if budget <= 0:
                return
            instr = dag.get_node_instr(node)
            qubits = instr.qubits

            if len(qubits) == 1 or isinstance(instr.operation, Barrier):
                continue
            elif len(qubits) == 2:
                q1, q2 = qubits

                if (
                    int(q2 in qubit_depends_on[q1]) + int(q1 in qubit_depends_on[q2])
                    == 1
                ):
                    dag.virtualize_node(node)
                    budget -= 1
                    continue

                to_add1 = Counter(qubit_depends_on[q2].keys()) + Counter([q2])
                to_add2 = Counter(qubit_depends_on[q1].keys()) + Counter([q1])

                qubit_depends_on[q1] += to_add1
                qubit_depends_on[q2] += to_add2

            elif len(qubits) > 2:
                raise ValueError("Cannot convert dag to qdg, too many qubits")

        # return qubit_depends_on

        # qubit_depends_on: dict[Qubit, dict[Qubit, set[int]]] = {
        #     qubit: {} for qubit in dag.qubits
        # }

        # def _update_dep(
        #     dep: dict[Qubit, set[int]], add_dep: dict[Qubit, set[int]]
        # ) -> None:
        #     for qubit, nodes in add_dep.items():
        #         if qubit not in dep:
        #             dep[qubit] = set()
        #         dep[qubit].update(nodes)

        # budget = self._max_vgates
        # nodes = list(nx.topological_sort(dag))
        # for node in nodes:
        #     if budget <= 0:
        #         return

        #     instr = dag.get_node_instr(node)
        #     qubits = instr.qubits

        #     if len(qubits) == 1 or isinstance(instr.operation, Barrier):
        #         continue
        #     elif len(qubits) == 2:
        #         q1, q2 = qubits

        #         to_add1 = qubit_depends_on[q2]
        #         _update_dep(to_add1, {q2: {node}})
        #         to_add2 = qubit_depends_on[q1]
        #         _update_dep(to_add2, {q1: {node}})

        #         _update_dep(qubit_depends_on[q1], to_add1)
        #         _update_dep(qubit_depends_on[q2], to_add2)

        #     elif len(qubits) > 2:
        #         raise ValueError("Cannot convert dag to qdg, too many qubits")

        # from pprint import pprint
        # pprint(qubit_depends_on)


class GreedyDependencyBreaker(ReduceQubitDependenciesCompiler):
    def __init__(self, max_vgates: int) -> None:
        self._max_vgates = max_vgates
        super().__init__()

    def _run_on_dag(self, dag: DAG) -> None:
        pass


class QubitDependencyMinimizer(ReduceQubitDependenciesCompiler):
    def __init__(self, max_vgates: int) -> None:
        self._max_vgates = max_vgates
        super().__init__()

    def _run_on_dag(self, dag: DAG) -> None:
        asp = dag_to_asp(dag)
        asp += self._min_dep_asp()

        symbols = get_optimal_symbols(asp)
        for symbol in symbols:
            if symbol.name != "vgate":
                continue
            gate_idx = symbol.arguments[0].number
            dag.virtualize_node(gate_idx)

    def _min_dep_asp(self) -> str:
        asp = f"""
        {{ vgate(Gate) }} :- gate_on_qubit(Gate, Qubit1), gate_on_qubit(Gate, Qubit2), Qubit1 != Qubit2.
        
        :- N = #count{{Gate : vgate(Gate)}}, N != {self._max_vgates}.
        
        path(Gate1, Gate2) :- wire(_, Gate1, Gate2), not vgate(Gate1), not vgate(Gate2).
        path(Gate1, Gate3) :- path(Gate1, Gate2), path(Gate2, Gate3).
        
        depends_on(Qubit1, Qubit2) :- 
            gate_on_qubit(Gate1, Qubit1),
            gate_on_qubit(Gate2, Qubit2),
            path(Gate1, Gate2),
            Qubit1 != Qubit2.
        
        num_deps(N) :- N = #count{{Qubit1, Qubit2 : depends_on(Qubit1, Qubit2)}}.
        
        :- wire(_, Gate1, Gate2), vgate(Gate1), vgate(Gate2).
        
        % num_vgates(N) :- N = #count{{Gate : vgate(Gate)}}.
        
        #minimize{{N : num_deps(N)}}.
        % # minimize{{N : num_vgates(N)}}.
        #show vgate/1.
        """
        return asp
