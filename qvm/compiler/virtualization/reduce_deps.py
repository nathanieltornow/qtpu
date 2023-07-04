import abc
from itertools import chain, combinations, product
from collections import Counter

import networkx as nx
from qiskit.circuit import QuantumCircuit, Qubit, Barrier

from qvm.compiler.asp import get_optimal_symbols, dag_to_asp
from qvm.compiler.types import CutCompiler
from qvm.compiler.dag import DAG


class QubitDependencyReducer(CutCompiler, abc.ABC):
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        dag = DAG(circuit)
        dag.compact()
        self._run_on_dag(dag)
        dag.fragment()
        return dag.to_circuit()

    @abc.abstractmethod
    def _run_on_dag(self, dag: DAG) -> None:
        ...


class CircularDependencyBreaker(QubitDependencyReducer):
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


class GreedyDependencyBreaker(QubitDependencyReducer):
    def __init__(self, max_vgates: int) -> None:
        self._max_vgates = max_vgates
        super().__init__()

    def _run_on_dag(self, dag: DAG) -> None:
        # we want to find out the nodes that connect the most qubit-pairs
        # for this, for each node, we find out which qubits depend on it
        # and with qubits the node depends on

        # these are the results we need
        op_influenced_by: dict[int, set[Qubit]] = {}
        op_influences: dict[int, set[Qubit]] = {}
        # # we keep track of the qubits influence by nodes
        qubit_depends_on: dict[Qubit, set[int]] = {qubit: set() for qubit in dag.qubits}

        budget = self._max_vgates
        two_qubit_nodes = set()
        nodes = list(nx.topological_sort(dag))
        for node in nodes:
            if budget <= 0:
                return

            instr = dag.get_node_instr(node)
            qubits = instr.qubits

            if len(qubits) == 1 or isinstance(instr.operation, Barrier):
                continue
            elif len(qubits) == 2:
                two_qubit_nodes.add(node)
                q1, q2 = qubits

                op_influences[node] = {q1, q2}
                op_influenced_by[node] = {q1, q2}

                # all operations the qubit 1 depends on
                for q1_dep_op in qubit_depends_on[q1]:
                    op_influenced_by[node].update(op_influenced_by[q1_dep_op])
                    op_influences[q1_dep_op].update({q1, q2})
                    qubit_depends_on[q2].add(q1_dep_op)

                for q2_dep_op in qubit_depends_on[q2]:
                    op_influenced_by[node].update(op_influenced_by[q2_dep_op])
                    op_influences[q2_dep_op].update({q1, q2})
                    qubit_depends_on[q1].add(q2_dep_op)

                qubit_depends_on[q1].add(node)
                qubit_depends_on[q2].add(node)

            elif len(qubits) > 2:
                raise ValueError("Cannot handle more than 2 qubits")

        instersect: dict[int, int] = {}
        for node in two_qubit_nodes:
            instersect[node] = len(op_influenced_by[node]) * len(
                op_influences[node]
            ) - len(op_influenced_by[node] & op_influences[node])

        sorted_nodes = sorted(
            instersect, key=lambda node: instersect[node], reverse=True
        )

        for node in sorted_nodes:
            if budget <= 0:
                return
            dag.virtualize_node(node)
            budget -= 1


class QubitDependencyMinimizer(QubitDependencyReducer):
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
        
        #minimize{{1000000@N : num_deps(N)}}.
        % #minimize{{N : num_vgates(N)}}.
        #show vgate/1.
        """
        return asp


def number_of_dependecies(dag: DAG) -> int:
    num_deps = 0
    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        qubits = instr.qubits
        if len(qubits) == 2:
            q1, q2 = qubits
            num_deps += len(dag.get_node_deps(node)) + 1
    return num_deps