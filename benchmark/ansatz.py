import itertools

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ZZFeatureMap, TwoLocal
from qiskit.circuit.random import random_circuit


def generate_ansatz(name: str, num_qubits: int, depth: int) -> QuantumCircuit:
    if name == "vqe":
        return linear_ansatz(num_qubits, depth)
    elif name == "tl":
        return brick_ansatz(num_qubits, depth)
    elif name == "zz":
        return zzfeaturemap_ansatz(num_qubits, depth)
    elif name == "rnd":
        return random_ansatz(num_qubits, depth)
    elif name == "clu":
        return cluster_ansatz(num_qubits, depth)
    else:
        raise ValueError(f"Unknown ansatz name: {name}")


def linear_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    circuit = TwoLocal(
        num_qubits, ["ry", "rz"], "rzz", entanglement="linear", reps=depth
    )
    return circuit.assign_parameters(
        {param: np.random.rand() * np.pi / 2 for param in circuit.parameters}
    ).decompose()


def brick_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    circuit = TwoLocal(
        num_qubits, ["ry", "rz"], "rzz", entanglement="pairwise", reps=depth
    )
    return circuit.assign_parameters(
        {param: np.random.rand() * np.pi / 2 for param in circuit.parameters}
    ).decompose()


def zzfeaturemap_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    circuit = ZZFeatureMap(num_qubits, reps=depth, entanglement="circular")
    return circuit.assign_parameters(
        {param: np.random.rand() * np.pi / 2 for param in circuit.parameters}
    ).decompose()


def random_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    return random_circuit(num_qubits, depth, max_operands=2)


def cluster_ansatz(num_qubits: int, depth: int, seed: int = None) -> QuantumCircuit:
    # fill clusters with random numbers between 15 - 20 qubits each, and the last cluster with the remaining qubits
    cluster_sizes = np.random.randint(15, 20, num_qubits // 20)
    cluster_sizes: list[int] = list(cluster_sizes) + [num_qubits - sum(cluster_sizes)]

    print(cluster_sizes)

    cluster_regs = [QuantumRegister(s, f"q_{i}") for i, s in enumerate(cluster_sizes)]
    circuit = QuantumCircuit(*cluster_regs)

    for _ in range(depth):
        for qreg in cluster_regs:
            circuit.compose(_cluster(qreg.size, 0.8, seed), qubits=qreg, inplace=True)

        for i in range(len(cluster_sizes) - 1):
            circuit.cx(cluster_regs[i][-1], cluster_regs[i + 1][0])
    return circuit


def _cluster(
    num_qubits: int, prob: float = 0.8, seed: int | None = None
) -> QuantumCircuit:

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    cluster = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        cluster.rz(rng.random() * np.pi / 2, i)
        cluster.ry(rng.random() * np.pi / 2, i)
    for q1, q2 in itertools.combinations(range(num_qubits), 2):
        if rng.random() < prob:
            cluster.cx(q1, q2)

    circuit = QuantumCircuit(num_qubits)
    circuit.append(cluster.to_gate(), range(num_qubits))
    return circuit
