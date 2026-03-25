"""
Quantum Kernel SVM Training Benchmark
=====================================

End-to-end training of a quantum kernel classifier using three approaches:

1. NAIVE: Generate and evaluate kernel circuits one-by-one
2. BATCH: Generate all kernel circuits, batch evaluate
3. HEINSUM (QTPU): Define HEinsum with ISwitches, train with PyTorch

All approaches train the SAME model:
    output = sigmoid(K[batch, support] @ W[support] + bias)
    
where K is the quantum kernel matrix (fidelity between data points).

The quantum kernel is FIXED (feature extractor), and W is TRAINABLE via PyTorch.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ClassicalRegister
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from evaluation.utils import log_result

from qtpu import HEinsum, QuantumTensor, CTensor, ISwitch, HEinsumContractor


# =============================================================================
# Configuration
# =============================================================================

NUM_QUBITS_LIST = [4, 6, 8]
NUM_SUPPORT_LIST = [10, 20, 30]
NUM_EPOCHS = 50
LEARNING_RATE = 0.1
NUM_TRAIN = 80
NUM_TEST = 20
SEED = 42


# =============================================================================
# Data
# =============================================================================


def get_data(n_train: int = NUM_TRAIN, n_test: int = NUM_TEST, seed: int = SEED):
    """Generate moons dataset for binary classification."""
    X, y = make_moons(n_samples=n_train + n_test, noise=0.1, random_state=seed)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X * np.pi / 2  # Scale to [0, pi] range
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=seed
    )
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# Circuit Helpers
# =============================================================================


def create_feature_map(num_qubits: int, x: np.ndarray, layers: int = 2) -> QuantumCircuit:
    """Create a feature map encoding data x into qubits."""
    qc = QuantumCircuit(num_qubits)
    n_features = len(x)
    
    for layer in range(layers):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            qc.ry(float(x[i % n_features]), i)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)
        # Add some ZZ interactions for expressivity
        for i in range(num_qubits):
            qc.rz(float(x[i % n_features]) * (layer + 1) * 0.3, i)
    
    return qc


def create_fidelity_circuit(x1: np.ndarray, x2: np.ndarray, num_qubits: int, layers: int = 2) -> QuantumCircuit:
    """Create fidelity circuit: U†(x2) U(x1) |0⟩."""
    qc = QuantumCircuit(num_qubits)
    
    # U(x1)
    qc.compose(create_feature_map(num_qubits, x1, layers), inplace=True)
    
    # U†(x2)
    qc.compose(create_feature_map(num_qubits, x2, layers).inverse(), inplace=True)
    
    return qc


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass
class TrainingResult:
    """Training result with timing and accuracy."""
    name: str
    
    # Timing
    kernel_compute_time: float = 0.0  # Time to compute kernel matrix
    training_time: float = 0.0        # Time for optimization loop
    total_time: float = 0.0
    
    # Per-epoch times
    epoch_times: list[float] = field(default_factory=list)
    
    # Accuracy
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    final_loss: float = 0.0
    
    # Info
    num_kernel_circuits: int = 0
    num_epochs: int = 0


# =============================================================================
# NAIVE Approach
# =============================================================================


def train_naive(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_support: np.ndarray,
    num_qubits: int,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> TrainingResult:
    """Naive approach: compute kernel one circuit at a time, then train."""
    result = TrainingResult(name="Naive (Qiskit)")
    
    n_train = len(X_train)
    n_test = len(X_test)
    n_support = len(X_support)
    
    estimator = StatevectorEstimator()
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])
    
    # Compute training kernel matrix K[train, support]
    kernel_start = perf_counter()
    
    K_train = np.zeros((n_train, n_support))
    for i in range(n_train):
        for j in range(n_support):
            qc = create_fidelity_circuit(X_train[i], X_support[j], num_qubits)
            job = estimator.run([(qc, observable)])
            K_train[i, j] = job.result()[0].data.evs
    
    # Compute test kernel matrix K[test, support]
    K_test = np.zeros((n_test, n_support))
    for i in range(n_test):
        for j in range(n_support):
            qc = create_fidelity_circuit(X_test[i], X_support[j], num_qubits)
            job = estimator.run([(qc, observable)])
            K_test[i, j] = job.result()[0].data.evs
    
    result.kernel_compute_time = perf_counter() - kernel_start
    result.num_kernel_circuits = (n_train + n_test) * n_support
    
    # Convert to probabilities
    K_train = (1 + K_train) / 2
    K_test = (1 + K_test) / 2
    
    # Convert to torch
    K_train_t = torch.tensor(K_train, dtype=torch.float64)
    K_test_t = torch.tensor(K_test, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    # Trainable weights
    W = nn.Parameter(torch.randn(n_support, dtype=torch.float64) * 0.1)
    bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    
    optimizer = torch.optim.Adam([W, bias], lr=lr)
    
    # Training loop
    train_start = perf_counter()
    
    for epoch in range(num_epochs):
        epoch_start = perf_counter()
        
        optimizer.zero_grad()
        
        logits = K_train_t @ W + bias
        probs = torch.sigmoid(logits)
        loss = nn.functional.binary_cross_entropy(probs, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        result.epoch_times.append(perf_counter() - epoch_start)
    
    result.training_time = perf_counter() - train_start
    result.total_time = result.kernel_compute_time + result.training_time
    result.num_epochs = num_epochs
    result.final_loss = loss.item()
    
    # Evaluate
    with torch.no_grad():
        train_preds = (torch.sigmoid(K_train_t @ W + bias) > 0.5).float()
        test_preds = (torch.sigmoid(K_test_t @ W + bias) > 0.5).float()
        result.train_accuracy = (train_preds == y_train_t).float().mean().item()
        result.test_accuracy = (test_preds == y_test_t).float().mean().item()
    
    return result


# =============================================================================
# BATCH Approach
# =============================================================================


def train_batch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_support: np.ndarray,
    num_qubits: int,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> TrainingResult:
    """Batch approach: generate all circuits, batch evaluate, then train."""
    result = TrainingResult(name="Batch (Qiskit)")
    
    n_train = len(X_train)
    n_test = len(X_test)
    n_support = len(X_support)
    
    estimator = StatevectorEstimator()
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])
    
    # Generate all circuits first
    kernel_start = perf_counter()
    
    train_circuits = []
    for i in range(n_train):
        for j in range(n_support):
            qc = create_fidelity_circuit(X_train[i], X_support[j], num_qubits)
            train_circuits.append(qc)
    
    test_circuits = []
    for i in range(n_test):
        for j in range(n_support):
            qc = create_fidelity_circuit(X_test[i], X_support[j], num_qubits)
            test_circuits.append(qc)
    
    # Batch evaluate
    train_jobs = [(qc, observable) for qc in train_circuits]
    train_results = estimator.run(train_jobs).result()
    K_train = np.array([r.data.evs for r in train_results]).reshape(n_train, n_support)
    
    test_jobs = [(qc, observable) for qc in test_circuits]
    test_results = estimator.run(test_jobs).result()
    K_test = np.array([r.data.evs for r in test_results]).reshape(n_test, n_support)
    
    result.kernel_compute_time = perf_counter() - kernel_start
    result.num_kernel_circuits = len(train_circuits) + len(test_circuits)
    
    # Convert to probabilities
    K_train = (1 + K_train) / 2
    K_test = (1 + K_test) / 2
    
    # Convert to torch
    K_train_t = torch.tensor(K_train, dtype=torch.float64)
    K_test_t = torch.tensor(K_test, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    # Trainable weights
    W = nn.Parameter(torch.randn(n_support, dtype=torch.float64) * 0.1)
    bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    
    optimizer = torch.optim.Adam([W, bias], lr=lr)
    
    # Training loop
    train_start = perf_counter()
    
    for epoch in range(num_epochs):
        epoch_start = perf_counter()
        
        optimizer.zero_grad()
        
        logits = K_train_t @ W + bias
        probs = torch.sigmoid(logits)
        loss = nn.functional.binary_cross_entropy(probs, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        result.epoch_times.append(perf_counter() - epoch_start)
    
    result.training_time = perf_counter() - train_start
    result.total_time = result.kernel_compute_time + result.training_time
    result.num_epochs = num_epochs
    result.final_loss = loss.item()
    
    # Evaluate
    with torch.no_grad():
        train_preds = (torch.sigmoid(K_train_t @ W + bias) > 0.5).float()
        test_preds = (torch.sigmoid(K_test_t @ W + bias) > 0.5).float()
        result.train_accuracy = (train_preds == y_train_t).float().mean().item()
        result.test_accuracy = (test_preds == y_test_t).float().mean().item()
    
    return result


# =============================================================================
# HEINSUM (QTPU) Approach
# =============================================================================


def train_heinsum(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_support: np.ndarray,
    num_qubits: int,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> TrainingResult:
    """HEinsum approach: define tensor network, train W via PyTorch autograd."""
    result = TrainingResult(name="HEinsum (QTPU)")
    
    n_train = len(X_train)
    n_test = len(X_test)
    n_support = len(X_support)
    
    # --- Build HEinsum for training data ---
    kernel_start = perf_counter()
    
    def build_kernel_heinsum(X_data: np.ndarray, name_prefix: str) -> HEinsum:
        """Build HEinsum for kernel matrix K[batch, support]."""
        n_batch = len(X_data)
        
        batch_param = Parameter(f"{name_prefix}_batch")
        support_param = Parameter("support")  # Shared across train/test
        
        def make_batch_circuit(idx: int) -> QuantumCircuit:
            return create_feature_map(num_qubits, X_data[idx])
        
        def make_support_circuit(idx: int) -> QuantumCircuit:
            return create_feature_map(num_qubits, X_support[idx]).inverse()
        
        qc = QuantumCircuit(num_qubits, num_qubits)
        qc.append(ISwitch(batch_param, num_qubits, n_batch, make_batch_circuit), range(num_qubits))
        qc.append(ISwitch(support_param, num_qubits, n_support, make_support_circuit), range(num_qubits))
        qc.measure(range(num_qubits), range(num_qubits))
        
        qtensor = QuantumTensor(qc)
        
        # Output: K[batch, support] - just the kernel matrix, no W yet
        heinsum = HEinsum(
            qtensors=[qtensor],
            ctensors=[],
            input_tensors=[],
            output_inds=(f"{name_prefix}_batch", "support"),
        )
        
        return heinsum
    
    # Build heinsums
    heinsum_train = build_kernel_heinsum(X_train, "train")
    heinsum_test = build_kernel_heinsum(X_test, "test")
    
    # Prepare contractors
    contractor_train = HEinsumContractor(heinsum_train)
    contractor_train.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    contractor_test = HEinsumContractor(heinsum_test)
    contractor_test.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    # Compute kernel matrices (quantum feature extraction)
    K_train_t = contractor_train.contract(input_tensors=[], circuit_params={})
    K_train_t = (1 + K_train_t) / 2  # Convert to probability
    K_train_t = K_train_t.detach()  # Detach - kernel is fixed
    
    K_test_t = contractor_test.contract(input_tensors=[], circuit_params={})
    K_test_t = (1 + K_test_t) / 2
    K_test_t = K_test_t.detach()
    
    result.kernel_compute_time = perf_counter() - kernel_start
    result.num_kernel_circuits = (n_train + n_test) * n_support
    
    # Targets
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    # Trainable weights
    W = nn.Parameter(torch.randn(n_support, dtype=torch.float64) * 0.1)
    bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    
    optimizer = torch.optim.Adam([W, bias], lr=lr)
    
    # Training loop (pure classical - fast!)
    train_start = perf_counter()
    
    for epoch in range(num_epochs):
        epoch_start = perf_counter()
        
        optimizer.zero_grad()
        
        logits = K_train_t @ W + bias
        probs = torch.sigmoid(logits)
        loss = nn.functional.binary_cross_entropy(probs, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        result.epoch_times.append(perf_counter() - epoch_start)
    
    result.training_time = perf_counter() - train_start
    result.total_time = result.kernel_compute_time + result.training_time
    result.num_epochs = num_epochs
    result.final_loss = loss.item()
    
    # Evaluate
    with torch.no_grad():
        train_preds = (torch.sigmoid(K_train_t @ W + bias) > 0.5).float()
        test_preds = (torch.sigmoid(K_test_t @ W + bias) > 0.5).float()
        result.train_accuracy = (train_preds == y_train_t).float().mean().item()
        result.test_accuracy = (test_preds == y_test_t).float().mean().item()
    
    return result


# =============================================================================
# Benchmark Runner
# =============================================================================


def run_comparison(
    num_qubits: int = 4,
    num_support: int = 20,
    num_epochs: int = NUM_EPOCHS,
):
    """Run comparison of all three approaches."""
    print("=" * 70)
    print(f"Quantum Kernel Training Comparison")
    print(f"  Qubits: {num_qubits}, Support vectors: {num_support}, Epochs: {num_epochs}")
    print("=" * 70)
    
    # Get data
    X_train, X_test, y_train, y_test = get_data()
    
    # Use first num_support training samples as support vectors
    X_support = X_train[:num_support]
    
    print(f"\nData: {len(X_train)} train, {len(X_test)} test, {num_support} support")
    print(f"Kernel circuits: {(len(X_train) + len(X_test)) * num_support}")
    
    results = []
    
    # 1. Naive
    print("\n[1] Naive (one-by-one)...")
    gc.collect()
    result_naive = train_naive(X_train, y_train, X_test, y_test, X_support, num_qubits, num_epochs)
    print(f"    Kernel time: {result_naive.kernel_compute_time:.2f}s")
    print(f"    Train time:  {result_naive.training_time:.3f}s")
    print(f"    Total:       {result_naive.total_time:.2f}s")
    print(f"    Test acc:    {result_naive.test_accuracy:.1%}")
    results.append(result_naive)
    
    # 2. Batch
    print("\n[2] Batch...")
    gc.collect()
    result_batch = train_batch(X_train, y_train, X_test, y_test, X_support, num_qubits, num_epochs)
    print(f"    Kernel time: {result_batch.kernel_compute_time:.2f}s")
    print(f"    Train time:  {result_batch.training_time:.3f}s")
    print(f"    Total:       {result_batch.total_time:.2f}s")
    print(f"    Test acc:    {result_batch.test_accuracy:.1%}")
    results.append(result_batch)
    
    # 3. HEinsum
    print("\n[3] HEinsum (QTPU)...")
    gc.collect()
    result_heinsum = train_heinsum(X_train, y_train, X_test, y_test, X_support, num_qubits, num_epochs)
    print(f"    Kernel time: {result_heinsum.kernel_compute_time:.2f}s")
    print(f"    Train time:  {result_heinsum.training_time:.3f}s")
    print(f"    Total:       {result_heinsum.total_time:.2f}s")
    print(f"    Test acc:    {result_heinsum.test_accuracy:.1%}")
    results.append(result_heinsum)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Kernel(s)':<12} {'Train(s)':<12} {'Total(s)':<12} {'Acc':<8}")
    print("-" * 64)
    for r in results:
        print(f"{r.name:<20} {r.kernel_compute_time:<12.2f} {r.training_time:<12.3f} {r.total_time:<12.2f} {r.test_accuracy:<8.1%}")
    
    print(f"\nSpeedups vs Naive:")
    print(f"  Batch kernel:   {result_naive.kernel_compute_time / result_batch.kernel_compute_time:.1f}x")
    print(f"  HEinsum kernel: {result_naive.kernel_compute_time / result_heinsum.kernel_compute_time:.1f}x")
    print(f"  HEinsum total:  {result_naive.total_time / result_heinsum.total_time:.1f}x")
    
    return results


# =============================================================================
# Benchmark Logging
# =============================================================================


def bench_naive(num_qubits: int, num_support: int) -> dict | None:
    """Benchmark naive approach."""
    print(f"Naive: qubits={num_qubits}, support={num_support}")
    X_train, X_test, y_train, y_test = get_data()
    X_support = X_train[:num_support]
    
    try:
        result = train_naive(X_train, y_train, X_test, y_test, X_support, num_qubits)
        return {
            "kernel_compute_time": result.kernel_compute_time,
            "training_time": result.training_time,
            "total_time": result.total_time,
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "final_loss": result.final_loss,
            "num_kernel_circuits": result.num_kernel_circuits,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def bench_batch(num_qubits: int, num_support: int) -> dict | None:
    """Benchmark batch approach."""
    print(f"Batch: qubits={num_qubits}, support={num_support}")
    X_train, X_test, y_train, y_test = get_data()
    X_support = X_train[:num_support]
    
    try:
        result = train_batch(X_train, y_train, X_test, y_test, X_support, num_qubits)
        return {
            "kernel_compute_time": result.kernel_compute_time,
            "training_time": result.training_time,
            "total_time": result.total_time,
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "final_loss": result.final_loss,
            "num_kernel_circuits": result.num_kernel_circuits,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def bench_heinsum(num_qubits: int, num_support: int) -> dict | None:
    """Benchmark HEinsum approach."""
    print(f"HEinsum: qubits={num_qubits}, support={num_support}")
    X_train, X_test, y_train, y_test = get_data()
    X_support = X_train[:num_support]
    
    try:
        result = train_heinsum(X_train, y_train, X_test, y_test, X_support, num_qubits)
        return {
            "kernel_compute_time": result.kernel_compute_time,
            "training_time": result.training_time,
            "total_time": result.total_time,
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "final_loss": result.final_loss,
            "num_kernel_circuits": result.num_kernel_circuits,
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
Quantum Kernel Training Benchmark

Usage: python kernel_training.py <command>

Commands:
    quick       Quick comparison (4 qubits, 10 support)
    compare     Full comparison with default settings
    naive       Run naive benchmark sweep
    batch       Run batch benchmark sweep
    heinsum     Run HEinsum benchmark sweep
    all         Run all benchmark sweeps
"""
    
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    def _run_sweep(bench_fn, log_path):
        for num_qubits in NUM_QUBITS_LIST:
            for num_support in NUM_SUPPORT_LIST:
                config = {"num_qubits": num_qubits, "num_support": num_support}
                print(f"  Config: {config}")
                result = bench_fn(num_qubits, num_support)
                log_result(log_path, config, result)

    if cmd == "quick":
        run_comparison(num_qubits=4, num_support=10, num_epochs=30)
    elif cmd == "compare":
        run_comparison(num_qubits=4, num_support=20, num_epochs=50)
    elif cmd == "naive":
        _run_sweep(bench_naive, "logs/hybrid_ml/kernel_naive.jsonl")
    elif cmd == "batch":
        _run_sweep(bench_batch, "logs/hybrid_ml/kernel_batch.jsonl")
    elif cmd == "heinsum":
        _run_sweep(bench_heinsum, "logs/hybrid_ml/kernel_heinsum.jsonl")
    elif cmd == "all":
        _run_sweep(bench_naive, "logs/hybrid_ml/kernel_naive.jsonl")
        _run_sweep(bench_batch, "logs/hybrid_ml/kernel_batch.jsonl")
        _run_sweep(bench_heinsum, "logs/hybrid_ml/kernel_heinsum.jsonl")
    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
