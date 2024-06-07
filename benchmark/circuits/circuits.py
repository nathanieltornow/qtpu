import os

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit


import numpy as np
import networkx as nx

from qiskit.algorithms.minimum_eigensolvers import VQE, QAOA
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, TwoLocal, QFT
from qiskit.primitives import Estimator, Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Maxcut


def get_circuits(benchname: str, qubit_range: tuple[int, int]) -> list[QuantumCircuit]:
    abs_path = os.path.dirname(os.path.abspath(__file__))
    circuits_dir = os.path.join(abs_path, benchname)
    files = [
        fname
        for fname in os.listdir(circuits_dir)
        if fname.endswith(".qasm")
        and fname.split(".")[0].isdigit()
        and int(fname.split(".")[0]) in range(*qubit_range)
    ]
    if len(files) == 0:
        raise ValueError("No circuits found for the given qubit range.")
    circuits = sorted(
        [
            QuantumCircuit.from_qasm_file(os.path.join(circuits_dir, fname))
            for fname in files
        ],
        key=lambda x: x.num_qubits,
    )
    dags = [circuit_to_dag(circ) for circ in circuits]
    for dag in dags:
        dag.remove_all_ops_named("barrier")
    return [dag_to_circuit(dag) for dag in dags]


def vqe(num_qubits: int, reps: int = 1) -> QuantumCircuit:
    qp = get_examplary_max_cut_qp(num_qubits)
    assert isinstance(qp, QuadraticProgram)

    ansatz = RealAmplitudes(num_qubits, reps=reps, entanglement="reverse_linear")
    vqe = VQE(ansatz=ansatz, optimizer=SLSQP(maxiter=25), estimator=Estimator())

    vqe._check_operator_ansatz(qp.to_ising()[0])
    qc = vqe.ansatz.bind_parameters(np.random.rand(vqe.ansatz.num_parameters))

    if num_qubits <= 16:
        qaoa_result = vqe.compute_minimum_eigenvalue(qp.to_ising()[0])
        qc = vqe.ansatz.bind_parameters(qaoa_result.optimal_point)

    # vqe_result = vqe.compute_minimum_eigenvalue(qp.to_ising()[0])
    # qc = vqe.ansatz.bind_parameters(vqe_result.optimal_point)

    _measure_all(qc)
    qc.name = "vqe"

    return qc.decompose().decompose()


def get_examplary_max_cut_qp(n_nodes: int, degree: int = 1) -> QuadraticProgram:
    """Returns a quadratic problem formulation of a max cut problem of a random graph.

    Keyword arguments:
    n_nodes -- number of graph nodes (and also number of qubits)
    degree -- edges per node
    """

    graph = nx.barabasi_albert_graph(n_nodes, degree)
    maxcut = Maxcut(graph)
    return maxcut.to_quadratic_program()


def _measure_all(circuit: QuantumCircuit) -> None:
    assert circuit.num_clbits == 0
    creg = ClassicalRegister(circuit.num_qubits)
    circuit.add_register(creg)
    circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))


if __name__ == "__main__":
    reps = 2
    for i in list(range(100, 1000, 50)) + list(range(1000, 2001, 100)):
        circ = vqe(i, reps)
        with open(f"bench/circuits/vqe_{reps}/{i}.qasm", "w") as f:
            f.write(circ.qasm())
