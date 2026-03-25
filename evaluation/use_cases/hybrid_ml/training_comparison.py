"""
Hybrid ML Training Benchmark: PennyLane vs QTPU
================================================

End-to-end training comparison on a binary classification task.

Task: Binary classification on make_moons dataset
Model: Variational Quantum Classifier with data re-uploading

Approaches:
1. PennyLane: Standard QNode with lightning.qubit backend
2. QTPU: HEinsum with ISwitch for batched data encoding

Metrics:
- Time per epoch
- Total training time
- Compilation/preparation overhead
- Final accuracy
"""

from __future__ import annotations

import gc
import tracemalloc
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation.utils import log_result

# PennyLane imports
import pennylane as qml

# QTPU imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ClassicalRegister

from qtpu import HEinsum, QuantumTensor, ISwitch, CTensor, HEinsumContractor


# =============================================================================
# Configuration
# =============================================================================

NUM_QUBITS_LIST = [4, 6, 8]
NUM_LAYERS_LIST = [2, 4]
NUM_EPOCHS_LIST = [5, 10, 20]
BATCH_SIZE = 32
LEARNING_RATE = 0.1
NUM_SAMPLES = 200  # Total dataset size


# =============================================================================
# Data Preparation
# =============================================================================


def get_data(n_samples: int = NUM_SAMPLES, seed: int = 42):
    """Generate make_moons dataset."""
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Scale to [0, pi] for encoding
    X = X * np.pi / 2
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


# =============================================================================
# PennyLane Implementation
# =============================================================================


def create_pennylane_model(n_qubits: int, n_layers: int, n_features: int = 2):
    """Create a PennyLane VQC model."""
    
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    # Use parameter-shift for fair comparison with QTPU
    # (adjoint differentiation is faster but not available on real hardware)
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        # Data re-uploading: encode data in each layer
        for layer in range(n_layers):
            # Data encoding
            for i in range(n_qubits):
                qml.RY(inputs[i % n_features], wires=i)
            
            # Trainable rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
            # Entanglement
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
        
        return qml.expval(qml.PauliZ(0))
    
    class PennyLaneVQC(nn.Module):
        def __init__(self):
            super().__init__()
            # Shape: (n_layers, n_qubits, 2) for RY and RZ
            self.weights = nn.Parameter(
                torch.randn(n_layers, n_qubits, 2) * 0.1
            )
        
        def forward(self, x):
            # Process batch
            outputs = []
            for xi in x:
                out = circuit(xi, self.weights)
                outputs.append(out)
            return torch.stack(outputs)
    
    return PennyLaneVQC()


def run_pennylane(
    n_qubits: int,
    n_layers: int,
    n_epochs: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Run PennyLane training."""
    gc.collect()
    
    n_features = X_train.shape[1]
    
    # Track memory during model creation
    tracemalloc.start()
    
    # Preparation: create model
    prep_start = perf_counter()
    model = create_pennylane_model(n_qubits, n_layers, n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    preparation_time = perf_counter() - prep_start
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Training
    epoch_times = []
    train_losses = []
    
    n_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for epoch in range(n_epochs):
        epoch_start = perf_counter()
        epoch_loss = 0.0
        
        # Shuffle data
        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(X_train))
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_time = perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        train_losses.append(epoch_loss / n_batches)
    
    total_training_time = sum(epoch_times)
    
    # Evaluation
    with torch.no_grad():
        test_outputs = model(X_test)
        test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
        accuracy = (test_preds == y_test).float().mean().item()
    
    return {
        "preparation_time": preparation_time,
        "total_training_time": total_training_time,
        "total_time": preparation_time + total_training_time,
        "epoch_times": epoch_times,
        "avg_epoch_time": np.mean(epoch_times),
        "first_epoch_time": epoch_times[0],
        "final_loss": train_losses[-1],
        "test_accuracy": accuracy,
        "peak_memory": peak_memory,
    }


# =============================================================================
# QTPU Implementation
# =============================================================================


def create_qtpu_feature_extractor(
    n_qubits: int,
    n_layers: int,
    X_data: np.ndarray,
    n_basis: int = 4,
) -> HEinsum:
    """Create a QTPU quantum feature extractor using HEinsum.
    
    Architecture:
    - ISwitch selects which data sample to encode (batch dimension)
    - ISwitch selects measurement basis (basis dimension)
    - No trainable quantum parameters - just feature extraction
    - Output: q[batch, basis] - quantum features for each sample
    
    Returns:
        HEinsum that outputs [batch, basis] tensor
    """
    n_features = X_data.shape[1]
    n_samples = X_data.shape[0]
    
    # Parameters for ISwitch dimensions
    batch_param = Parameter("batch")
    basis_param = Parameter("basis")
    
    def make_circuit(idx: int) -> QuantumCircuit:
        """Create feature encoding circuit for sample idx."""
        qc = QuantumCircuit(n_qubits)
        x = X_data[idx]
        
        for layer in range(n_layers):
            # Data encoding
            for i in range(n_qubits):
                qc.ry(float(x[i % n_features]), i)
            
            # Fixed entanglement
            for i in range(n_qubits - 1):
                qc.cz(i, i + 1)
            
            # More encoding with interaction
            for i in range(n_qubits):
                qc.rz(float(x[i % n_features]) * (layer + 1) * 0.5, i)
        
        return qc
    
    def make_basis_circuit(k: int) -> QuantumCircuit:
        """Create measurement basis rotation for basis k."""
        qc = QuantumCircuit(n_qubits)
        angle = k * np.pi / n_basis
        for i in range(n_qubits):
            qc.ry(angle + i * 0.1, i)
        return qc
    
    # Main circuit with ISwitches for batch and basis
    qc = QuantumCircuit(n_qubits)
    qc.add_register(ClassicalRegister(n_qubits))
    
    # Data encoding (batch dimension)
    qc.append(ISwitch(batch_param, n_qubits, n_samples, make_circuit), range(n_qubits))
    
    # Measurement basis (basis dimension)  
    qc.append(ISwitch(basis_param, n_qubits, n_basis, make_basis_circuit), range(n_qubits))
    
    qc.measure(range(n_qubits), range(n_qubits))
    
    qtensor = QuantumTensor(qc)
    
    # HEinsum: output has both batch and basis dimensions
    heinsum = HEinsum(
        qtensors=[qtensor],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch", "basis"),
    )
    
    return heinsum


def run_qtpu(
    n_qubits: int,
    n_layers: int,
    n_epochs: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Run QTPU training: quantum feature extraction + classical training.
    
    1. Extract quantum features once (no gradient through quantum)
    2. Train classical weights using standard PyTorch autograd (fast!)
    """
    gc.collect()
    
    n_basis = 4  # Number of measurement bases
    
    # Track memory during model creation
    tracemalloc.start()
    
    # Preparation: create HEinsum and extract features
    prep_start = perf_counter()
    
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()
    
    # Create feature extractors
    heinsum_train = create_qtpu_feature_extractor(n_qubits, n_layers, X_train_np, n_basis)
    heinsum_test = create_qtpu_feature_extractor(n_qubits, n_layers, X_test_np, n_basis)
    
    # Prepare contractors
    contractor_train = HEinsumContractor(heinsum_train)
    contractor_train.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    contractor_test = HEinsumContractor(heinsum_test)
    contractor_test.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    # Extract quantum features (once, no gradients needed)
    q_train = contractor_train.contract(input_tensors=[], circuit_params={}).detach()
    q_test = contractor_test.contract(input_tensors=[], circuit_params={}).detach()
    
    preparation_time = perf_counter() - prep_start
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Classical trainable weights: q[batch, basis] @ W[basis] + bias -> logits[batch]
    W = torch.randn(n_basis, dtype=torch.float64) * 0.1
    W = nn.Parameter(W)
    bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    
    n_params = W.numel() + bias.numel()
    
    optimizer = torch.optim.Adam([W, bias], lr=LEARNING_RATE)
    
    # Targets
    y_train_t = y_train.to(torch.float64)
    y_test_t = y_test.to(torch.float64)
    
    # Training loop (pure classical - fast!)
    epoch_times = []
    train_losses = []
    
    for epoch in range(n_epochs):
        epoch_start = perf_counter()
        
        optimizer.zero_grad()
        
        # Forward: q[batch, basis] @ W[basis] + bias -> logits[batch]
        logits = q_train @ W + bias
        
        # Binary cross-entropy loss
        probs = torch.sigmoid(logits)
        loss = nn.functional.binary_cross_entropy(probs, y_train_t)
        
        # Backward (pure classical autograd - fast!)
        loss.backward()
        optimizer.step()
        
        epoch_time = perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        train_losses.append(loss.item())
    
    total_training_time = sum(epoch_times)
    
    # Evaluation
    with torch.no_grad():
        test_logits = q_test @ W + bias
        test_probs = torch.sigmoid(test_logits)
        test_binary = (test_probs > 0.5).float()
        accuracy = (test_binary == y_test_t).float().mean().item()
    
    return {
        "preparation_time": preparation_time,
        "total_training_time": total_training_time,
        "total_time": preparation_time + total_training_time,
        "epoch_times": epoch_times,
        "avg_epoch_time": np.mean(epoch_times),
        "first_epoch_time": epoch_times[0],
        "final_loss": train_losses[-1],
        "test_accuracy": accuracy,
        "peak_memory": peak_memory,
        "num_params": n_params,
        "quantum_feature_shape": list(q_train.shape),
        "num_circuits_represented": int(np.prod(heinsum_train.quantum_tensors[0].shape)) if heinsum_train.quantum_tensors[0].shape else 1,
    }


# =============================================================================
# Benchmark Functions
# =============================================================================


def bench_pennylane(n_qubits: int, n_layers: int, n_epochs: int) -> dict | None:
    """Benchmark PennyLane training."""
    print(f"PennyLane: qubits={n_qubits}, layers={n_layers}, epochs={n_epochs}")
    
    X_train, y_train, X_test, y_test = get_data()
    
    try:
        result = run_pennylane(
            n_qubits, n_layers, n_epochs,
            X_train, y_train, X_test, y_test
        )
        return result
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def bench_qtpu(n_qubits: int, n_layers: int, n_epochs: int) -> dict | None:
    """Benchmark QTPU training."""
    print(f"QTPU: qubits={n_qubits}, layers={n_layers}, epochs={n_epochs}")
    
    X_train, y_train, X_test, y_test = get_data()
    
    try:
        result = run_qtpu(
            n_qubits, n_layers, n_epochs,
            X_train, y_train, X_test, y_test
        )
        return result
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
Hybrid ML Training Benchmark: PennyLane vs QTPU

Usage: python training_comparison.py <command>

Commands:
    pennylane   Run PennyLane benchmark
    qtpu        Run QTPU benchmark  
    all         Run all benchmarks
    quick       Quick test (small config)

Configuration:
    Qubits: {NUM_QUBITS_LIST}
    Layers: {NUM_LAYERS_LIST}
    Epochs: {NUM_EPOCHS_LIST}
    Batch size: {BATCH_SIZE}
    Dataset size: {NUM_SAMPLES}
""".format(
        NUM_QUBITS_LIST=NUM_QUBITS_LIST,
        NUM_LAYERS_LIST=NUM_LAYERS_LIST,
        NUM_EPOCHS_LIST=NUM_EPOCHS_LIST,
        BATCH_SIZE=BATCH_SIZE,
        NUM_SAMPLES=NUM_SAMPLES,
    )
    
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    def _run_sweep(bench_fn, log_path):
        for n_qubits in NUM_QUBITS_LIST:
            for n_layers in NUM_LAYERS_LIST:
                for n_epochs in NUM_EPOCHS_LIST:
                    config = {"n_qubits": n_qubits, "n_layers": n_layers, "n_epochs": n_epochs}
                    print(f"  Config: {config}")
                    result = bench_fn(n_qubits, n_layers, n_epochs)
                    log_result(log_path, config, result)

    if cmd == "pennylane":
        _run_sweep(bench_pennylane, "logs/hybrid_ml/pennylane_training.jsonl")
    elif cmd == "qtpu":
        _run_sweep(bench_qtpu, "logs/hybrid_ml/qtpu_training.jsonl")
    elif cmd == "all":
        print("Running all benchmarks...")
        _run_sweep(bench_pennylane, "logs/hybrid_ml/pennylane_training.jsonl")
        _run_sweep(bench_qtpu, "logs/hybrid_ml/qtpu_training.jsonl")
    elif cmd == "quick":
        # Quick test with minimal config
        print("Quick test...")
        X_train, y_train, X_test, y_test = get_data(n_samples=50)
        
        print("\n--- PennyLane ---")
        pl_result = run_pennylane(4, 2, 3, X_train, y_train, X_test, y_test)
        print(f"Prep time: {pl_result['preparation_time']:.3f}s")
        print(f"Training time: {pl_result['total_training_time']:.3f}s")
        print(f"Accuracy: {pl_result['test_accuracy']:.2%}")
        
        print("\n--- QTPU ---")
        qtpu_result = run_qtpu(4, 2, 3, X_train, y_train, X_test, y_test)
        print(f"Prep time: {qtpu_result['preparation_time']:.3f}s")
        print(f"Training time: {qtpu_result['total_training_time']:.3f}s")
        print(f"Accuracy: {qtpu_result['test_accuracy']:.2%}")
        
        print("\n--- Comparison ---")
        print(f"Prep speedup: {pl_result['preparation_time'] / qtpu_result['preparation_time']:.2f}x")
        print(f"Training speedup: {pl_result['total_training_time'] / qtpu_result['total_training_time']:.2f}x")
        print(f"Total speedup: {pl_result['total_time'] / qtpu_result['total_time']:.2f}x")
    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
