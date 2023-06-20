from qiskit.transpiler import CouplingMap

from qvm.dag import DAG


def apply_virtual_qubit_routing(
    dag: DAG, coupling_map: CouplingMap, initial_layout: list[int]
) -> None:
    pass
