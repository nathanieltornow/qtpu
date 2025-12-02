"""
Hybrid ML Inference Benchmark: Timing Breakdown Comparison
==========================================================

Compares three approaches for quantum kernel inference in hybrid ML:

1. NAIVE LOOP:
   For each (batch, support) pair: generate circuit → transpile ONE-BY-ONE → store
   This is the worst-case pattern where transpilation happens sequentially for each
   circuit, preventing any parallelization benefits.

2. BATCH (Optimized):
   Generate ALL circuits first → BATCH transpile (Qiskit can parallelize) → store
   This is the standard optimization: batch transpilation can leverage parallel
   processing and common substructure caching.

3. HEINSUM (QTPU):
   Define HEinsum with ISwitches → Compile once → Broadcast execute
   Our approach: circuits are defined symbolically and compiled efficiently,
   with the tensor network structure enabling maximum sharing.

Metrics tracked:
- Preparation time (CPU): Circuit generation + compilation/transpilation
- Quantum time (QPU): Estimated QPU execution time (using FakeMarrakesh scheduling)
- Classical time (GPU/CPU): Post-processing and tensor contraction

NOTE: This benchmark does NOT actually execute quantum circuits. Instead:
- For Naive/Batch: Estimates QPU time using Qiskit's scheduling on FakeMarrakesh
- For HEinsum: Uses FakeQPUCudaQBackend which estimates QPU time

All transpilation uses FakeMarrakesh backend with optimization_level=3.

Sweeps:
- Circuit sizes (qubits): 4, 6, 8, 10, 12
- Feature dimensions: 2, 4, 8
- Batch sizes: 10, 50, 100, 200
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ClassicalRegister

import benchkit as bk
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from qtpu.core import HEinsum, QuantumTensor, CTensor, ISwitch
from qtpu.runtime import HEinsumRuntime, FakeQPUCudaQBackend

from evaluation.analysis import estimate_runtime

# Shared backend for transpilation
_FAKE_BACKEND = None


def get_fake_backend():
    """Lazily initialize the fake backend for transpilation."""
    global _FAKE_BACKEND
    if _FAKE_BACKEND is None:
        _FAKE_BACKEND = FakeMarrakesh()
    return _FAKE_BACKEND


# =============================================================================
# Configuration
# =============================================================================

CIRCUIT_SIZES = [4, 6, 8, 10, 12]
FEATURE_DIMS = [2, 4, 8]
BATCH_SIZES = [10, 50, 100, 200]

# Fixed support vector count (typical for kernel methods)
NUM_SUPPORT = 20

# Number of ansatz layers
NUM_LAYERS = 2


# =============================================================================
# Circuit Construction Helpers
# =============================================================================


def create_feature_map(
    num_qubits: int, x: np.ndarray, layers: int = 2
) -> QuantumCircuit:
    """Create a hardware-efficient feature map encoding data into qubits.

    Architecture:
    - H gates on all qubits
    - RY rotations encoding features (cycling through feature dims)
    - CZ entanglement ladder
    - Repeated for `layers` times

    Args:
        num_qubits: Number of qubits in the circuit.
        x: Feature vector to encode.
        layers: Number of encoding layers.

    Returns:
        QuantumCircuit with the feature map.
    """
    qc = QuantumCircuit(num_qubits)
    num_features = len(x)

    for layer in range(layers):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            # Cycle through features
            qc.ry(float(x[i % num_features]), i)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    return qc


def create_fidelity_circuit(
    x1: np.ndarray,
    x2: np.ndarray,
    num_qubits: int,
    layers: int = 2,
) -> QuantumCircuit:
    """Create a fidelity circuit U†(x2) U(x1) |0⟩.

    The expectation value <0|U†(x2) U(x1)|0> gives the kernel value.

    Args:
        x1: First feature vector (batch sample).
        x2: Second feature vector (support vector).
        num_qubits: Number of qubits.
        layers: Number of encoding layers.

    Returns:
        QuantumCircuit computing the fidelity.
    """
    qc = QuantumCircuit(num_qubits)
    num_features = len(x1)

    # U(x1) - encode first sample
    for layer in range(layers):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            qc.ry(float(x1[i % num_features]), i)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    # U†(x2) - inverse encoding of second sample
    for layer in range(layers):
        for i in range(num_qubits - 2, -1, -1):
            qc.cz(i, i + 1)
        for i in range(num_qubits - 1, -1, -1):
            qc.ry(-float(x2[i % num_features]), i)
        for i in range(num_qubits - 1, -1, -1):
            qc.h(i)

    return qc


# =============================================================================
# NAIVE Approach
# =============================================================================


def run_naive(
    X_batch: np.ndarray,
    X_support: np.ndarray,
    W: np.ndarray,
    num_qubits: int,
    layers: int = 2,
) -> dict:
    """Naive approach: generate and transpile circuits ONE AT A TIME.

    This is the worst-case pattern where each circuit is:
    1. Generated
    2. Transpiled individually (no batching)
    3. Submitted for execution

    The key overhead is that transpilation happens sequentially for each circuit,
    preventing any parallelization or caching benefits.
    """
    n_batch = len(X_batch)
    n_support = len(X_support)
    total_circuits = n_batch * n_support

    backend = get_fake_backend()

    # Preparation phase: generate and transpile circuits ONE BY ONE
    # This is the naive pattern - no batching of transpilation
    prep_start = perf_counter()

    transpiled_circuits = []
    for b in range(n_batch):
        for s in range(n_support):
            # Generate circuit
            qc = create_fidelity_circuit(X_batch[b], X_support[s], num_qubits, layers)
            # Transpile ONE circuit at a time (inefficient!)
            transpiled = transpile(
                qc,
                backend=backend,
                optimization_level=3,
            )
            transpiled_circuits.append(transpiled)

    preparation_time = perf_counter() - prep_start

    # Estimate quantum execution time using analysis.estimate_runtime
    # This uses Qiskit's scheduling on FakeMarrakesh to get realistic QPU times
    quantum_time = estimate_runtime(transpiled_circuits)

    # Classical post-processing time estimate (minimal for naive approach)
    # Just the kernel computation + matmul
    classical_start = perf_counter()
    K = np.random.randn(n_batch, n_support)  # Dummy values
    K = (1 + K) / 2
    output = K @ W
    classical_time = perf_counter() - classical_start

    return {
        "preparation_time": preparation_time,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "total_time": preparation_time + quantum_time + classical_time,
        "num_circuits": total_circuits,
    }


# =============================================================================
# BATCH Approach
# =============================================================================


def run_batch(
    X_batch: np.ndarray,
    X_support: np.ndarray,
    W: np.ndarray,
    num_qubits: int,
    layers: int = 2,
) -> dict:
    """Batch approach: generate ALL circuits first, then BATCH transpile.

    This is the optimized pattern:
    1. Generate all fidelity circuits (fast, just Python)
    2. Batch transpile all circuits at once (Qiskit can parallelize)
    3. Estimate QPU execution time

    The key benefit is that batch transpilation can leverage parallel processing
    and caching, making it much faster than naive one-by-one transpilation.
    """
    n_batch = len(X_batch)
    n_support = len(X_support)

    backend = get_fake_backend()

    # Preparation phase: generate all circuits first
    prep_start = perf_counter()

    # Step 1: Generate all circuits (fast)
    circuits = []
    for b in range(n_batch):
        for s in range(n_support):
            qc = create_fidelity_circuit(X_batch[b], X_support[s], num_qubits, layers)
            circuits.append(qc)

    # Step 2: BATCH transpile all circuits at once
    # This allows Qiskit to parallelize and cache common substructures
    transpiled_circuits = [
        transpile(
            circuit,
            backend=backend,
            optimization_level=3,
        )
        for circuit in circuits
    ]

    preparation_time = perf_counter() - prep_start

    # Estimate quantum execution time using analysis.estimate_runtime
    # This uses Qiskit's scheduling on FakeMarrakesh to get realistic QPU times
    quantum_time = estimate_runtime(transpiled_circuits)

    # Classical post-processing time estimate
    classical_start = perf_counter()
    K = np.random.randn(n_batch, n_support)  # Dummy values
    K = (1 + K) / 2
    output = K @ W
    classical_time = perf_counter() - classical_start

    return {
        "preparation_time": preparation_time,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "total_time": preparation_time + quantum_time + classical_time,
        "num_circuits": len(transpiled_circuits),
    }


# =============================================================================
# HEINSUM (QTPU) Approach
# =============================================================================


def run_heinsum(
    X_batch: np.ndarray,
    X_support: np.ndarray,
    W: np.ndarray,
    num_qubits: int,
    layers: int = 2,
) -> dict:
    """HEinsum approach: define tensor network with ISwitches, estimate QPU time.

    1. Create QuantumTensor with ISwitches for batch/support dimensions
    2. Define HEinsum contraction with classical weight tensor
    3. Compile using FakeQPUCudaQBackend (estimates QPU time, no actual execution)
    4. Classical contraction is fused into HEinsum
    """
    n_batch = len(X_batch)
    n_support = len(X_support)

    # Preparation phase: create HEinsum structure
    prep_start = perf_counter()

    # Build circuit with ISwitches
    qc = QuantumCircuit(num_qubits)
    qc.add_register(ClassicalRegister(num_qubits))

    batch_param = Parameter("batch")
    support_param = Parameter("support")

    def make_batch_circuit(idx: int) -> QuantumCircuit:
        return create_feature_map(num_qubits, X_batch[idx], layers)

    def make_support_circuit(idx: int) -> QuantumCircuit:
        return create_feature_map(num_qubits, X_support[idx], layers).inverse()

    batch_iswitch = ISwitch(batch_param, num_qubits, n_batch, make_batch_circuit)
    support_iswitch = ISwitch(
        support_param, num_qubits, n_support, make_support_circuit
    )

    qc.append(batch_iswitch, range(num_qubits))
    qc.append(support_iswitch, range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    # Create quantum tensor
    qtensor = QuantumTensor(qc)

    # Create classical weight tensor
    W_tensor = CTensor(torch.tensor(W, dtype=torch.float64), inds=("support",))

    # Define HEinsum: contract quantum kernel with weights
    heinsum = HEinsum(
        qtensors=[qtensor],
        ctensors=[W_tensor],
        input_tensors=[],
        output_inds=("batch",),
    )

    # Create runtime with FakeQPUCudaQBackend (estimates QPU time)
    backend = FakeQPUCudaQBackend(shots=1000)
    runtime = HEinsumRuntime(heinsum, backend=backend, dtype=torch.float64)
    runtime.prepare(optimize=False)

    preparation_time = perf_counter() - prep_start

    # Execute (uses FakeQPUCudaQBackend which estimates QPU time)
    exec_start = perf_counter()
    output_tensor, timing = runtime.execute()
    quantum_time = timing.quantum_estimated_qpu_time

    # Classical post-processing (minimal for HEinsum - contraction already done)
    classical_start = perf_counter()
    output = output_tensor.detach().cpu().numpy()
    output = (1 + output) / 2  # Convert to probability scale
    classical_time = perf_counter() - classical_start

    return {
        "preparation_time": preparation_time,
        "quantum_time": quantum_time,
        "classical_time": classical_time,
        "classical_contraction_time": timing.classical_contraction_time,
        "total_time": preparation_time + quantum_time + classical_time,
        "num_circuits": timing.num_circuits,
    }


# =============================================================================
# Benchmark Functions with BenchKit Logging
# =============================================================================


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(feature_dim=FEATURE_DIMS)
@bk.foreach(batch_size=BATCH_SIZES)
@bk.log("logs/hybrid_ml/naive_breakdown.jsonl")
def bench_naive(circuit_size: int, feature_dim: int, batch_size: int) -> dict | None:
    """Benchmark naive (sequential) approach."""
    print(f"Naive: qubits={circuit_size}, features={feature_dim}, batch={batch_size}")

    # Generate random data
    np.random.seed(42)
    X_batch = np.random.randn(batch_size, feature_dim) * np.pi
    X_support = np.random.randn(NUM_SUPPORT, feature_dim) * np.pi
    W = np.random.randn(NUM_SUPPORT) * 0.1

    try:
        result = run_naive(X_batch, X_support, W, circuit_size, NUM_LAYERS)
        return {
            **result,
            "num_support": NUM_SUPPORT,
            "num_layers": NUM_LAYERS,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(feature_dim=FEATURE_DIMS)
@bk.foreach(batch_size=BATCH_SIZES)
@bk.log("logs/hybrid_ml/batch_breakdown.jsonl")
def bench_batch(circuit_size: int, feature_dim: int, batch_size: int) -> dict | None:
    """Benchmark batch approach."""
    print(f"Batch: qubits={circuit_size}, features={feature_dim}, batch={batch_size}")

    np.random.seed(42)
    X_batch = np.random.randn(batch_size, feature_dim) * np.pi
    X_support = np.random.randn(NUM_SUPPORT, feature_dim) * np.pi
    W = np.random.randn(NUM_SUPPORT) * 0.1

    try:
        result = run_batch(X_batch, X_support, W, circuit_size, NUM_LAYERS)
        return {
            **result,
            "num_support": NUM_SUPPORT,
            "num_layers": NUM_LAYERS,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(feature_dim=FEATURE_DIMS)
@bk.foreach(batch_size=BATCH_SIZES)
@bk.log("logs/hybrid_ml/heinsum_breakdown.jsonl")
def bench_heinsum(circuit_size: int, feature_dim: int, batch_size: int) -> dict | None:
    """Benchmark HEinsum (QTPU) approach."""
    print(f"HEinsum: qubits={circuit_size}, features={feature_dim}, batch={batch_size}")

    np.random.seed(42)
    X_batch = np.random.randn(batch_size, feature_dim) * np.pi
    X_support = np.random.randn(NUM_SUPPORT, feature_dim) * np.pi
    W = np.random.randn(NUM_SUPPORT) * 0.1

    try:
        result = run_heinsum(X_batch, X_support, W, circuit_size, NUM_LAYERS)
        return {
            **result,
            "num_support": NUM_SUPPORT,
            "num_layers": NUM_LAYERS,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import sys

    usage = """
Hybrid ML Inference Benchmark

Usage: python run.py <command>

Commands:
    naive       Run naive (sequential) benchmark
    batch       Run batch benchmark
    heinsum     Run HEinsum (QTPU) benchmark
    all         Run all benchmarks

Configuration:
    Circuit sizes: {CIRCUIT_SIZES}
    Feature dims:  {FEATURE_DIMS}
    Batch sizes:   {BATCH_SIZES}
    Support vecs:  {NUM_SUPPORT}
    Layers:        {NUM_LAYERS}
""".format(
        CIRCUIT_SIZES=CIRCUIT_SIZES,
        FEATURE_DIMS=FEATURE_DIMS,
        BATCH_SIZES=BATCH_SIZES,
        NUM_SUPPORT=NUM_SUPPORT,
        NUM_LAYERS=NUM_LAYERS,
    )

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "naive":
        bench_naive()
    elif cmd == "batch":
        bench_batch()
    elif cmd == "heinsum":
        bench_heinsum()
    elif cmd == "all":
        print("Running all benchmarks...")
        bench_naive()
        bench_batch()
        bench_heinsum()
    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
