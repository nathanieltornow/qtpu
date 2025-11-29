"""Benchmark circuit generators for QTPU evaluation.

Includes standard MQT benchmarks and distributed quantum computing benchmarks
inspired by "Variational Quantum Eigensolvers in the Era of Distributed Quantum
Computers" (Khait et al., 2023, arXiv:2302.14067).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes


def create_distributed_vqe(
    num_qubits: int,
    qubits_per_cluster: int = 10,
    layers: int = 3,
    inter_cluster_gates: int = 1,
    seed: int | None = None,
) -> QuantumCircuit:
    """Create a distributed VQE ansatz for multi-module quantum computers.

    Based on the architecture from Khait et al. (2023): dense entanglement within
    each cluster (module), sparse inter-cluster connections.

    Args:
        num_qubits: Total number of qubits.
        qubits_per_cluster: Qubits per cluster/module. Clusters are created to
            cover all qubits (last cluster may be smaller).
        layers: Number of ansatz layers.
        inter_cluster_gates: Number of CX gates between adjacent clusters per layer.
        seed: Random seed for parameter initialization.

    Returns:
        QuantumCircuit with distributed VQE ansatz.
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate cluster structure
    num_full_clusters = num_qubits // qubits_per_cluster
    remainder = num_qubits % qubits_per_cluster

    cluster_sizes = [qubits_per_cluster] * num_full_clusters
    if remainder > 0:
        cluster_sizes.append(remainder)

    num_clusters = len(cluster_sizes)

    # Build circuit
    qc = QuantumCircuit(num_qubits)

    for layer in range(layers):
        # Intra-cluster: dense all-to-all entanglement (hard to simulate classically)
        qubit_idx = 0
        for cluster_id, cluster_size in enumerate(cluster_sizes):
            # Rotation layer (Ry, Rz on each qubit)
            for i in range(cluster_size):
                qc.ry(np.random.random() * 2 * np.pi, qubit_idx + i)
                qc.rz(np.random.random() * 2 * np.pi, qubit_idx + i)

            # All-to-all entanglement within cluster
            for i in range(cluster_size):
                for j in range(i + 1, cluster_size):
                    qc.cx(qubit_idx + i, qubit_idx + j)

            qubit_idx += cluster_size

        # Inter-cluster: sparse connections (few gates between adjacent clusters)
        if layer < layers - 1:  # No inter-cluster on last layer
            qubit_idx = 0
            for cluster_id in range(num_clusters - 1):
                cluster_size = cluster_sizes[cluster_id]
                next_cluster_start = qubit_idx + cluster_size

                # Add inter_cluster_gates CX gates between clusters
                for g in range(min(inter_cluster_gates, cluster_size)):
                    q1 = qubit_idx + cluster_size - 1 - g  # End of current cluster
                    q2 = next_cluster_start + g  # Start of next cluster
                    if q2 < num_qubits:
                        qc.cx(q1, q2)

                qubit_idx += cluster_size

    return qc


def get_benchmark(
    bench: str, circuit_size: int, cluster_size: int = 10
) -> QuantumCircuit:
    """Retrieve benchmark circuit by name and size.

    Args:
        bench: Benchmark name. Options:
            - "qnn", "wstate", "vqe_su2": MQT standard benchmarks
            - "dist-vqe": Distributed VQE ansatz (Khait et al. style)
            - "dist-qml": Distributed QML ansatz
        circuit_size: Number of qubits.

    Returns:
        QuantumCircuit for the benchmark.
    """
    from mqt.bench import get_benchmark_indep

    if bench == "qnn":
        return get_benchmark_indep("qnn", circuit_size=circuit_size)
    elif bench == "wstate":
        return get_benchmark_indep("wstate", circuit_size=circuit_size)
    elif bench == "vqe_su2":
        return get_benchmark_indep("vqe_su2", circuit_size=circuit_size)
    elif bench == "dist-vqe":
        # Distributed VQE: variable cluster size, 3 layers, 3 inter-cluster gates
        return create_distributed_vqe(
            num_qubits=circuit_size,
            qubits_per_cluster=cluster_size,
            layers=3,
            inter_cluster_gates=3,
            seed=42,
        )
    else:
        raise ValueError(f"Unknown benchmark: {bench}")
