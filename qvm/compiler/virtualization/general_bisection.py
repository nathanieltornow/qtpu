import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection as bisect
from qiskit.circuit import QuantumCircuit, Qubit, Barrier

from qvm.compiler._types import VirtualizationCompiler
from qvm.compiler.dag import DAG, dag_to_qcg
from qvm.virtual_gates import VIRTUAL_GATE_TYPES


class GeneralBisectionCompiler(VirtualizationCompiler):
    def __init__(
        self,
        max_virtual_gates: int = 3,
        reverse_order: bool = True,
        optimal_bisection: bool = False,
    ) -> None:
        self._max_virtual_gates = max_virtual_gates
        self._reverse_order = reverse_order
        self._optimal_bisection = optimal_bisection

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        dag = DAG(circuit)
        qcg = dag_to_qcg(dag)
        A, B = bisect(qcg)
        vgates_inserted = self._virtualize_between_qubit_sets(
            dag, A, B, self._max_virtual_gates
        )
        partitions = [A, B]
        while vgates_inserted < self._max_virtual_gates:
            largest_fragment = max(partitions, key=lambda f: len(f))
            partitions.remove(largest_fragment)
            if len(largest_fragment) == 1:
                partitions += [largest_fragment]
                break
            A, B = bisect(qcg.subgraph(largest_fragment))
            vgates_inserted += self._virtualize_between_qubit_sets(
                dag, A, B, self._max_virtual_gates - vgates_inserted
            )
            partitions += [A, B]

        dag.fragment()
        return dag.to_circuit()

    def _virtualize_between_qubit_sets(
        self, dag: DAG, qubits1: set[Qubit], qubits2: set[Qubit], vgate_limit: int
    ) -> int:
        num_new_vgates = 0
        sorted_nodes = list(nx.topological_sort(dag))
        if self._reverse_order:
            sorted_nodes = sorted_nodes[::-1]
        for node in sorted_nodes:
            instr = dag.get_node_instr(node)
            qubits = instr.qubits
            if len(qubits) == 1 or isinstance(instr.operation, Barrier):
                continue
            elif len(qubits) > 2:
                raise ValueError(f"Instruction acts on more than two qubits: {instr}")
            elif len(qubits) == 2:
                qubit1, qubit2 = instr.qubits
                if (qubit1 in qubits1 and qubit2 in qubits2) or (
                    qubit1 in qubits2 and qubit2 in qubits1
                ):
                    instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](
                        instr.operation
                    )
                    num_new_vgates += 1
                    vgate_limit -= 1
                    if vgate_limit == 0:
                        break
        return num_new_vgates
