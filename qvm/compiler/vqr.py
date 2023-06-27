from qiskit.circuit import Qubit
from qiskit.transpiler import CouplingMap

from .util import virtualize_between_qubit_pairs
from .dag import DAG, dag_to_qcg


def delay_swapping(
    dag: DAG,
    coupling_map: CouplingMap,
    initial_layout: list[int],
    vgate_limit: int = -1,
) -> None:
    qubit_map = {dag.qubits[i]: p for i, p in enumerate(initial_layout)}
    qcg = dag_to_qcg(dag)
    qubit_pairs: set[tuple[Qubit, Qubit]] = set()

    for qubit1, qubit2 in qcg.edges:
        if coupling_map.distance(qubit_map[qubit1], qubit_map[qubit2]) > 1:
            qubit_pairs.add((qubit1, qubit2))

    virtualize_between_qubit_pairs(dag, qubit_pairs, vgate_limit)
