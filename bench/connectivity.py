import networkx as nx

from qiskit.circuit import QuantumCircuit

from qvm.util import circuit_to_qcg


def average_degree(cricuit: QuantumCircuit) -> float:
    qvg = circuit_to_qcg(cricuit, use_qubit_idx=True)
    return sum([d for _, d in qvg.degree(weight="weight")]) / qvg.number_of_nodes()


from circuits.qft import qft
from circuits.two_local import two_local
from circuits.qaoa import qaoa
from circuits.vqe import vqe

print(average_degree(two_local(10, 1, "linear")))