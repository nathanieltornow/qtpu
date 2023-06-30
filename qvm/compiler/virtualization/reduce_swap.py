import networkx as nx
from qiskit.circuit import QuantumCircuit, Qubit, Barrier
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit.providers import BackendV2

from qvm.compiler._types import VirtualizationCompiler
from qvm.compiler.dag import DAG, dag_to_qcg
from qvm.virtual_gates import VIRTUAL_GATE_TYPES


class ReduceSWAPCompiler(VirtualizationCompiler):
    def __init__(
        self,
        backend: BackendV2,
        max_virtual_gates: int = 3,
        reverse_order: bool = False,
        initial_layout: list[int] | None = None,
    ) -> None:
        self._backend = backend
        self._max_virtual_gates = max_virtual_gates
        self._reverse_order = reverse_order
        self._initial_layout = initial_layout

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        init_layout = self._initial_layout
        if init_layout is None:
            init_layout = _get_initial_layout(circuit, self._backend)

        try:
            coupling_map = self._backend.coupling_map
        except AttributeError:
            coupling_map = CouplingMap.from_full(self._backend.num_qubits)

        dag = DAG(circuit)
        self._run_on_dag(dag, coupling_map, init_layout)
        dag.fragment()
        return dag.to_circuit()

    def _run_on_dag(
        self, dag: DAG, coupling_map: CouplingMap, initial_layout: Qubit
    ) -> None:
        sorted_nodes = list(nx.topological_sort(dag))
        if self._reverse_order:
            sorted_nodes = sorted_nodes[::-1]

        budget = self._max_virtual_gates
        if budget < 0:
            budget = dag.number_of_nodes()
        elif budget == 0:
            return

        qubit_mapping = dict(zip(dag.qubits, initial_layout))

        max_distance = 0
        qcg = dag_to_qcg(dag)
        max_distance = max(
            coupling_map.distance(qubit_mapping[q1], qubit_mapping[q2])
            for q1, q2 in qcg.edges
        )
        print(max_distance)

        while max_distance > 1:
            for node in sorted_nodes:
                if budget == 0:
                    break
                instr = dag.get_node_instr(node)
                if len(instr.qubits) == 1 or isinstance(instr.operation, Barrier):
                    continue
                elif len(instr.qubits) > 2:
                    raise ValueError(
                        f"Instruction acts on more than two qubits: {instr}"
                    )
                elif len(instr.qubits) == 2:
                    p1, p2 = tuple(qubit_mapping[qubit] for qubit in instr.qubits)
                    if coupling_map.distance(p1, p2) >= max_distance:
                        instr.operation = VIRTUAL_GATE_TYPES[instr.operation.name](
                            instr.operation
                        )
                        budget -= 1

            max_distance -= 1


def _get_initial_layout(circuit: QuantumCircuit, backend: BackendV2) -> list[int]:
    t_circuit = transpile(circuit, backend, optimization_level=3)
    return _initial_layout_from_transpiled_circuit(circuit, t_circuit)


def _initial_layout_from_transpiled_circuit(
    original_circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit
) -> list[int]:
    if transpiled_circuit._layout is None:
        raise ValueError("Circuit has no layout.")
    initial_layout = [0] * original_circuit.num_qubits
    layout = transpiled_circuit._layout.initial_layout.get_virtual_bits()
    for i, qubit in enumerate(original_circuit.qubits):
        initial_layout[i] = layout[qubit]
    return initial_layout
