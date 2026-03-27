"""
Hybrid ML Inference Benchmark: Timing Breakdown Comparison
==========================================================

Compares three approaches for quantum kernel inference in hybrid ML:

1. NAIVE:
   Expand ISwitches into individual circuits, process ONE BY ONE.
   Each circuit is code-generated for CUDA-Q individually.

2. BATCH:
   Expand ISwitches into individual circuits, process ALL AT ONCE.
   All circuits are code-generated for CUDA-Q in a batch.

3. HEINSUM (QTPU):
   Define HEinsum with ISwitches → Compile once → Broadcast execute.
   Single circuit with ISwitches, compiled once to CUDA-Q.

All three approaches start from the same HEinsum definition, ensuring
a fair comparison of the execution strategies.

Metrics tracked:
- Preparation time (CPU): Circuit generation + CUDA-Q code generation
- Quantum time (QPU): Estimated QPU execution time
- Classical time (GPU/CPU): Post-processing and tensor contraction

Sweeps:
- Circuit sizes (qubits): 20, 50, 100
- Feature dimensions: 2, 4, 8
- Batch sizes: 10, 50, 100, 200
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ClassicalRegister

from evaluation.utils import log_result

from qtpu.core import HEinsum, QuantumTensor, CTensor, ISwitch
from qtpu.runtime.baseline import run_naive, run_batch, run_heinsum


# =============================================================================
# Configuration
# =============================================================================

CIRCUIT_SIZES = [20, 50, 100]
FEATURE_DIMS = [2, 4, 8]
BATCH_SIZES = [10, 50, 100, 200]
SEEDS = [42, 43, 44]

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
    """
    qc = QuantumCircuit(num_qubits)
    num_features = len(x)

    for layer in range(layers):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            qc.ry(float(x[i % num_features]), i)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    return qc


def build_heinsum(
    X_batch: np.ndarray,
    X_support: np.ndarray,
    W: np.ndarray,
    num_qubits: int,
    layers: int = 2,
) -> HEinsum:
    """Build a HEinsum for quantum kernel computation.
    
    This creates a HEinsum with:
    - One QuantumTensor with ISwitches for batch/support dimensions
    - One classical weight tensor
    - Output contracted over support dimension
    
    Args:
        X_batch: Batch feature vectors (n_batch, feature_dim).
        X_support: Support feature vectors (n_support, feature_dim).
        W: Support weights (n_support,).
        num_qubits: Number of qubits in the circuit.
        layers: Number of encoding layers.
        
    Returns:
        HEinsum ready for execution.
    """
    n_batch = len(X_batch)
    n_support = len(X_support)
    
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
    support_iswitch = ISwitch(support_param, num_qubits, n_support, make_support_circuit)

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
    
    return heinsum


# =============================================================================
# Benchmark Functions
# =============================================================================


def bench_naive(circuit_size: int, feature_dim: int, batch_size: int, seed: int = 42) -> dict | None:
    """Benchmark naive (sequential) approach."""
    print(f"Naive: qubits={circuit_size}, features={feature_dim}, batch={batch_size}, seed={seed}")

    np.random.seed(seed)
    X_batch = np.random.randn(batch_size, feature_dim) * np.pi
    X_support = np.random.randn(NUM_SUPPORT, feature_dim) * np.pi
    W = np.random.randn(NUM_SUPPORT) * 0.1

    try:
        heinsum = build_heinsum(X_batch, X_support, W, circuit_size, NUM_LAYERS)
        _, timing = run_naive(heinsum, include_codegen=True)
        
        return {
            "preparation_time": timing.circuit_compilation_time,
            "quantum_time": timing.quantum_estimated_qpu_time,
            "num_circuits": timing.num_circuits,
            "total_code_lines": timing.total_code_lines,
            "classical_time": timing.classical_contraction_time,
            "total_time": timing.total_time,
            "num_support": NUM_SUPPORT,
            "num_layers": NUM_LAYERS,
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def bench_batch(circuit_size: int, feature_dim: int, batch_size: int, seed: int = 42) -> dict | None:
    """Benchmark batch approach."""
    print(f"Batch: qubits={circuit_size}, features={feature_dim}, batch={batch_size}, seed={seed}")

    np.random.seed(seed)
    X_batch = np.random.randn(batch_size, feature_dim) * np.pi
    X_support = np.random.randn(NUM_SUPPORT, feature_dim) * np.pi
    W = np.random.randn(NUM_SUPPORT) * 0.1

    try:
        heinsum = build_heinsum(X_batch, X_support, W, circuit_size, NUM_LAYERS)
        _, timing = run_batch(heinsum, include_codegen=True)
        
        return {
            "preparation_time": timing.circuit_compilation_time,
            "quantum_time": timing.quantum_estimated_qpu_time,
            "num_circuits": timing.num_circuits,
            "total_code_lines": timing.total_code_lines,
            "classical_time": timing.classical_contraction_time,
            "total_time": timing.total_time,
            "num_support": NUM_SUPPORT,
            "num_layers": NUM_LAYERS,
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def bench_heinsum(circuit_size: int, feature_dim: int, batch_size: int, seed: int = 42) -> dict | None:
    """Benchmark HEinsum (QTPU) approach."""
    print(f"HEinsum: qubits={circuit_size}, features={feature_dim}, batch={batch_size}, seed={seed}")

    np.random.seed(seed)
    X_batch = np.random.randn(batch_size, feature_dim) * np.pi
    X_support = np.random.randn(NUM_SUPPORT, feature_dim) * np.pi
    W = np.random.randn(NUM_SUPPORT) * 0.1

    try:
        heinsum = build_heinsum(X_batch, X_support, W, circuit_size, NUM_LAYERS)
        
        # Use skip_execution=True to get timing estimate without 
        # actually running the simulation (FakeQPUCudaQBackend)
        _, timing = run_heinsum(
            heinsum, 
            skip_execution=True,
        )
        
        return {
            "preparation_time": timing.circuit_compilation_time,
            "quantum_time": timing.quantum_estimated_qpu_time,
            "num_circuits": timing.num_circuits,
            "total_code_lines": timing.total_code_lines,
            "classical_time": timing.classical_contraction_time,
            "total_time": timing.total_time,
            "num_support": NUM_SUPPORT,
            "num_layers": NUM_LAYERS,
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
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

    def _run_sweep(bench_fn, log_path):
        for circuit_size in CIRCUIT_SIZES:
            for feature_dim in FEATURE_DIMS:
                for batch_size in BATCH_SIZES:
                    for seed in SEEDS:
                        config = {"circuit_size": circuit_size, "feature_dim": feature_dim, "batch_size": batch_size, "seed": seed}
                        print(f"  Config: {config}")
                        result = bench_fn(circuit_size, feature_dim, batch_size, seed=seed)
                        log_result(log_path, config, result)

    if cmd == "naive":
        _run_sweep(bench_naive, "logs/hybrid_ml/naive_breakdown.jsonl")
    elif cmd == "batch":
        _run_sweep(bench_batch, "logs/hybrid_ml/batch_breakdown.jsonl")
    elif cmd == "heinsum":
        _run_sweep(bench_heinsum, "logs/hybrid_ml/heinsum_breakdown.jsonl")
    elif cmd == "all":
        print("Running all benchmarks...")
        _run_sweep(bench_naive, "logs/hybrid_ml/naive_breakdown.jsonl")
        _run_sweep(bench_batch, "logs/hybrid_ml/batch_breakdown.jsonl")
        _run_sweep(bench_heinsum, "logs/hybrid_ml/heinsum_breakdown.jsonl")
    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
