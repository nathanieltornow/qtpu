import random

import numpy as np
import networkx as nx

from qiskit.algorithms.minimum_eigensolvers import VQE, QAOA
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, TwoLocal, QFT
from qiskit.primitives import Estimator, Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Maxcut


def get_circuits(
    benchname: str, param: int | float, nums_qubits: list[int] | None = None
) -> list[QuantumCircuit]:
    """Returns a list of quantum circuits for a specific benchmark."""
    if nums_qubits is None:
        nums_qubits = [4, 6, 8, 10, 12, 14]

    if benchname == "2local":
        return [two_local(n, param) for n in nums_qubits]
    elif benchname == "qaoa":
        return [qaoa(n, param) for n in nums_qubits]
    elif benchname == "qft":
        return [qft(n, n - param) for n in nums_qubits]
    elif benchname == "dj":
        return [dj(n) for n in nums_qubits]
    elif benchname == "vqe":
        return [vqe(n, param) for n in nums_qubits]
    elif benchname == "hamsim":
        return [hamsim(n, param) for n in nums_qubits]
    elif benchname == "ghz":
        return [ghz(n) for n in nums_qubits]
    raise ValueError(f"Unknown benchmark name: {benchname}")


def _measure_all(circuit: QuantumCircuit) -> None:
    assert circuit.num_clbits == 0
    creg = ClassicalRegister(circuit.num_qubits)
    circuit.add_register(creg)
    circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))


def vqe(num_qubits: int, reps: int = 1) -> QuantumCircuit:
    qp = get_examplary_max_cut_qp(num_qubits)
    assert isinstance(qp, QuadraticProgram)

    ansatz = RealAmplitudes(num_qubits, reps=reps, entanglement="circular")
    vqe = VQE(ansatz=ansatz, optimizer=SLSQP(maxiter=25), estimator=Estimator())

    vqe_result = vqe.compute_minimum_eigenvalue(qp.to_ising()[0])
    qc = vqe.ansatz.bind_parameters(vqe_result.optimal_point)

    qc.measure_all()
    qc.name = "vqe"

    return qc.decompose().decompose()


def two_local(
    num_qubits: int, reps: int = 1, entanglement: str = "circular"
) -> QuantumCircuit:
    np.random.seed(100)
    qc = TwoLocal(
        num_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="rzz",
        entanglement=entanglement,
        reps=reps,
    )
    num_params = qc.num_parameters
    qc = qc.bind_parameters(np.random.rand(num_params))
    _measure_all(qc)
    qc.name = "twolacalrandom"

    return qc.decompose()


def qft(num_qubits: int, approx: int = 0) -> QuantumCircuit:
    circuit = QFT(num_qubits, approximation_degree=approx, do_swaps=False)
    circuit.measure_all()
    return circuit.decompose()


def qaoa(num_qubits: int, deg: int):
    G = nx.powerlaw_cluster_graph(num_qubits, deg, 0.1)

    if True:
        import matplotlib.pyplot as plt

        nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")

        # Show the plot
        plt.show()

    nqubits = len(G.nodes())
    qc = QuantumCircuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    gamma = np.random.uniform(0, 2 * np.pi, 1)[0]
    beta = np.random.uniform(0, np.pi, 1)[0]

    for pair in sorted(G.edges(), key=lambda x: (x[0], x[1])):
        qc.rzz(gamma, pair[0], pair[1])
    # mixer unitary
    for i in range(0, nqubits):
        qc.rx(beta, i)

    qc.measure_all()

    return qc


# def qaoa(num_qubits: int, degree: int = 1, reps: int = 1) -> QuantumCircuit:
#     """Returns a quantum circuit implementing the Quantum Approximation Optimization Algorithm for a specific max-cut
#      example.

#     Keyword arguments:
#     num_qubits -- number of qubits of the returned quantum circuit
#     """

#     qp = get_examplary_max_cut_qp(num_qubits, degree=degree)
#     assert isinstance(qp, QuadraticProgram)

#     qaoa = QAOA(sampler=Sampler(), reps=reps, optimizer=SLSQP(maxiter=25))

#     qaoa._check_operator_ansatz(qp.to_ising()[0])

#     qc = qaoa.ansatz.bind_parameters(np.random.rand(qaoa.ansatz.num_parameters))

#     if num_qubits < 1:
#         qaoa_result = qaoa.compute_minimum_eigenvalue(qp.to_ising()[0])
#         qc = qaoa.ansatz.bind_parameters(qaoa_result.optimal_point)

#     qc.name = "qaoa"

#     return qc.decompose().decompose()


def hamsim(num_qubits: int, total_time: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    hbar = 0.658212  # eV*fs
    jz = (
        hbar * np.pi / 4
    )  # eV, coupling coeff; Jz<0 is antiferromagnetic, Jz>0 is ferromagnetic
    freq = 0.0048  # 1/fs, frequency of MoSe2 phonon

    w_ph = 2 * np.pi * freq
    e_ph = 3 * np.pi * hbar / (8 * np.cos(np.pi * freq))

    for step in range(total_time):
        # Simulate the Hamiltonian term-by-term
        t = step + 0.5

        # Single qubit terms
        psi = -2.0 * e_ph * np.cos(w_ph * t) * 1 / hbar
        for i in range(num_qubits):
            circuit.h(i)
            circuit.rz(psi, i)
            circuit.h(i)

        # Coupling terms
        psi2 = -2.0 * jz * 1 / hbar
        for i in range(num_qubits - 1):
            circuit.rzz(psi2, i, i + 1)

    circuit.measure_all()
    return circuit


def ghz(num_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    circuit.cx(range(num_qubits - 1), range(1, num_qubits))
    circuit.measure_all()
    return circuit


def dj_oracle(case: str, n: int) -> QuantumCircuit:
    # plus one output qubit
    oracle_qc = QuantumCircuit(n + 1)

    if case == "balanced":
        np.random.seed(10)
        b_str = ""
        for _ in range(n):
            b = np.random.randint(0, 2)
            b_str = b_str + str(b)

        for qubit in range(len(b_str)):
            if b_str[qubit] == "1":
                oracle_qc.x(qubit)

        for qubit in range(n):
            oracle_qc.cx(qubit, n)

        for qubit in range(len(b_str)):
            if b_str[qubit] == "1":
                oracle_qc.x(qubit)

    if case == "constant":
        output = np.random.randint(2)
        if output == 1:
            oracle_qc.x(n)

    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Oracle"  # To show when we display the circuit
    return oracle_gate


def dj_algorithm(oracle: QuantumCircuit, n: int) -> QuantumCircuit:
    dj_circuit = QuantumCircuit(n + 1, n)

    dj_circuit.x(n)
    dj_circuit.h(n)

    for qubit in range(n):
        dj_circuit.h(qubit)

    dj_circuit.append(oracle, range(n + 1))

    for qubit in range(n):
        dj_circuit.h(qubit)

    dj_circuit.barrier()
    for i in range(n):
        dj_circuit.measure(i, i)

    return dj_circuit


def dj(n: int, balanced: bool = True) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Deutsch-Josza algorithm.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    balanced -- True for a balanced and False for a constant oracle
    """

    oracle_mode = "balanced" if balanced else "constant"
    n = n - 1  # because of ancilla qubit
    oracle_gate = dj_oracle(oracle_mode, n)
    qc = dj_algorithm(oracle_gate, n)
    qc.name = "dj"

    return qc


def get_examplary_max_cut_qp(n_nodes: int, degree: int = 1) -> QuadraticProgram:
    """Returns a quadratic problem formulation of a max cut problem of a random graph.

    Keyword arguments:
    n_nodes -- number of graph nodes (and also number of qubits)
    degree -- edges per node
    """

    graph = nx.barabasi_albert_graph(n_nodes, degree)
    maxcut = Maxcut(graph)
    return maxcut.to_quadratic_program()


if __name__ == "__main__":
    import os
    import itertools
    from qiskit.compiler import transpile

    layers = [1, 2, 3]
    num_qubits = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

    for layer, nq in itertools.product(layers, num_qubits):
        circuit = transpile(vqe(nq, layer), basis_gates=["cx", "sx", "rz", "x"], optimization_level=3)
        qasm = circuit.qasm()

        file_name = f"bench/qasm/vqe_{layer}_{nq}.qasm"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, "w") as f:
            f.write(qasm)
