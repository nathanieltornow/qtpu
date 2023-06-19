from qiskit.circuit import Qubit

from qvm.dag import DAG


def virtualize_between_qubitsets(dag: DAG, qubit_groups: list[set[Qubit]]):
    pass


def virtualize_qubit_connection(dag: DAG, qubit1: Qubit, qubit2: Qubit):
    pass


def break_qubit_dependecy(dag: DAG, qubit1: Qubit, qubit2: Qubit):
    pass
