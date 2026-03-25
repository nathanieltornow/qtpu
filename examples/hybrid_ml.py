"""Hybrid quantum-classical ML model with qTPU (paper Listing 2).

Builds a quantum kernel over a batch of data points using ISwitch,
pairs it with a classical weight tensor, and contracts via HEinsum.
This is a minimal hybrid tensor network (hTN) for classification.

Usage:
    uv run python examples/hybrid_ml.py
"""

import numpy as np
import torch
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

from qtpu import CTensor, HEinsum, HEinsumRuntime, ISwitch, QuantumTensor


# -- Helpers ----------------------------------------------------------------

def make_feature_circuit(features: np.ndarray) -> QuantumCircuit:
    """Encode a single feature vector into a 4-qubit circuit."""
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)
    for i, f in enumerate(features[:n_qubits]):
        qc.ry(float(f), i)
    # Entangling layer
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def build_quantum_kernel(
    data: np.ndarray, support_vectors: np.ndarray
) -> tuple[QuantumTensor, list[str]]:
    """Build a qTensor whose axes span batch and support-vector dimensions.

    Each (batch_idx, support_idx) selects a circuit that computes the
    kernel entry K(x_i, s_j) via an ISwitch.
    """
    n_batch, n_support = len(data), len(support_vectors)
    n_qubits = 4

    # ISwitch over batch dimension
    batch_param = Parameter("batch")
    batch_iswitch = ISwitch(
        batch_param, n_qubits, n_batch,
        selector=lambda i, _d=data: make_feature_circuit(_d[i]),
    )

    # ISwitch over support-vector dimension
    support_param = Parameter("support")
    support_iswitch = ISwitch(
        support_param, n_qubits, n_support,
        selector=lambda j, _s=support_vectors: make_feature_circuit(_s[j]).inverse(),
    )

    # Compose: encode data, then apply inverse of support vector encoding
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.append(batch_iswitch, range(n_qubits))
    qc.append(support_iswitch, range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    return QuantumTensor(qc), ["batch", "support"]


def main():
    np.random.seed(42)

    # -- 1. Synthetic data: 8 samples, 3 support vectors, 4 features -------
    n_batch, n_support, n_features = 8, 3, 4
    data = np.random.randn(n_batch, n_features)
    support_vectors = np.random.randn(n_support, n_features)
    alpha = np.random.randn(n_support)  # SVM-style dual weights

    # -- 2. Build the hybrid tensor network ---------------------------------
    #    qTensor K[batch, support] : quantum kernel matrix
    #    cTensor alpha[support]    : classical weight vector
    #    HEinsum: sum_support K[batch, support] * alpha[support] -> pred[batch]
    qtensor, _ = build_quantum_kernel(data, support_vectors)
    ctensor = CTensor(alpha, inds=("support",))

    heinsum = HEinsum(
        qtensors=[qtensor],
        ctensors=[ctensor],
        input_tensors=[],
        output_inds=("batch",),  # output has batch dimension
    )

    print(f"qTensor shape : {qtensor.shape}  (batch x support)")
    print(f"cTensor shape : {ctensor.shape}  (support,)")
    print(f"Einsum expr   : {heinsum.einsum_expr}")

    # -- 3. Execute ---------------------------------------------------------
    runtime = HEinsumRuntime(heinsum, backend="cudaq")
    runtime.prepare()
    result, timing = runtime.execute()

    predictions = torch.sign(result)
    print(f"\nRaw scores : {result}")
    print(f"Predictions: {predictions}")
    print(f"Circuits   : {timing.num_circuits}")
    print(f"Total time : {timing.total_time:.3f}s")


if __name__ == "__main__":
    main()
