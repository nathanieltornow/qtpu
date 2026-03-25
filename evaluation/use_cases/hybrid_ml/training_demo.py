"""
QTPU Training Demo
==================

End-to-end training demonstration showing QTPU can train quantum ML models.

Architecture:
- Quantum feature extractor (fixed circuits with ISwitch for batching)
- Classical trainable weights (CTensor)
- Standard PyTorch optimization

Shows:
- Training curves (loss, accuracy vs epoch)
- Final test accuracy
- Scalability with different configurations
"""

from __future__ import annotations

import gc
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_circles, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation.utils import log_result

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ClassicalRegister

from qtpu import HEinsum, QuantumTensor, ISwitch, CTensor, HEinsumContractor


# =============================================================================
# Configuration
# =============================================================================

DATASETS = ["moons", "circles", "iris"]
NUM_QUBITS_LIST = [4, 6]
NUM_LAYERS_LIST = [2, 3]
NUM_EPOCHS = 100
LEARNING_RATE = 0.1
N_BASIS = 4


# =============================================================================
# Data Preparation
# =============================================================================


def get_dataset(name: str, n_samples: int = 200, seed: int = 42):
    """Load a dataset for binary classification."""
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=seed)
    elif name == "iris":
        # Binary: setosa vs non-setosa
        data = load_iris()
        X, y = data.data[:, :2], (data.target == 0).astype(int)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Scale to reasonable range for rotation angles
    X = X * np.pi / 3
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# QTPU Model
# =============================================================================


def create_quantum_features(
    X_data: np.ndarray,
    n_qubits: int,
    n_layers: int,
    n_basis: int = N_BASIS,
) -> HEinsum:
    """Create quantum feature extractor with HEinsum.
    
    Output: q[batch, basis] - quantum features for each sample
    """
    n_features = X_data.shape[1]
    n_samples = X_data.shape[0]
    
    batch_param = Parameter("batch")
    basis_param = Parameter("basis")
    
    def make_encoding_circuit(idx: int) -> QuantumCircuit:
        """Feature encoding for sample idx."""
        qc = QuantumCircuit(n_qubits)
        x = X_data[idx]
        
        for layer in range(n_layers):
            # Angle encoding
            for i in range(n_qubits):
                qc.ry(float(x[i % n_features]), i)
            
            # Entanglement
            for i in range(n_qubits - 1):
                qc.cz(i, i + 1)
            if n_qubits > 2:
                qc.cz(n_qubits - 1, 0)  # Ring topology
            
            # Feature interaction
            for i in range(n_qubits):
                angle = float(x[i % n_features]) * (layer + 1) * 0.3
                qc.rz(angle, i)
        
        return qc
    
    def make_basis_circuit(k: int) -> QuantumCircuit:
        """Measurement basis k."""
        qc = QuantumCircuit(n_qubits)
        angle = k * np.pi / n_basis
        for i in range(n_qubits):
            qc.ry(angle + i * 0.2, i)
        return qc
    
    # Build circuit
    qc = QuantumCircuit(n_qubits)
    qc.add_register(ClassicalRegister(n_qubits))
    qc.append(ISwitch(batch_param, n_qubits, n_samples, make_encoding_circuit), range(n_qubits))
    qc.append(ISwitch(basis_param, n_qubits, n_basis, make_basis_circuit), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    
    qtensor = QuantumTensor(qc)
    
    return HEinsum(
        qtensors=[qtensor],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch", "basis"),
    )


def train_qtpu(
    dataset: str,
    n_qubits: int,
    n_layers: int,
    n_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    n_basis: int = N_BASIS,
    verbose: bool = True,
) -> dict:
    """Train QTPU model and return training history."""
    gc.collect()
    
    # Load data
    X_train, X_test, y_train, y_test = get_dataset(dataset)
    n_train, n_test = len(X_train), len(X_test)
    
    if verbose:
        print(f"\nDataset: {dataset} (train={n_train}, test={n_test})")
        print(f"Config: qubits={n_qubits}, layers={n_layers}, basis={n_basis}")
    
    # Create quantum feature extractors
    prep_start = perf_counter()
    
    heinsum_train = create_quantum_features(X_train, n_qubits, n_layers, n_basis)
    heinsum_test = create_quantum_features(X_test, n_qubits, n_layers, n_basis)
    
    contractor_train = HEinsumContractor(heinsum_train)
    contractor_train.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    contractor_test = HEinsumContractor(heinsum_test)
    contractor_test.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    # Extract quantum features (once)
    q_train = contractor_train.contract([], {}).detach()
    q_test = contractor_test.contract([], {}).detach()
    
    prep_time = perf_counter() - prep_start
    
    if verbose:
        print(f"Quantum features: train={q_train.shape}, test={q_test.shape}")
        print(f"Preparation time: {prep_time:.2f}s")
    
    # Classical trainable weights
    W = nn.Parameter(torch.randn(n_basis, dtype=torch.float64) * 0.1)
    bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    
    optimizer = torch.optim.Adam([W, bias], lr=lr)
    
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    # Training history
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "epoch_time": [],
    }
    
    # Training loop
    train_start = perf_counter()
    
    for epoch in range(n_epochs):
        epoch_start = perf_counter()
        
        optimizer.zero_grad()
        
        # Forward
        logits = q_train @ W + bias
        probs = torch.sigmoid(logits)
        loss = nn.functional.binary_cross_entropy(probs, y_train_t)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        epoch_time = perf_counter() - epoch_start
        
        # Metrics
        with torch.no_grad():
            train_preds = (probs > 0.5).float()
            train_acc = (train_preds == y_train_t).float().mean().item()
            
            test_logits = q_test @ W + bias
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs > 0.5).float()
            test_acc = (test_preds == y_test_t).float().mean().item()
        
        history["epoch"].append(epoch)
        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)
        
        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, train_acc={train_acc:.2%}, test_acc={test_acc:.2%}")
    
    train_time = perf_counter() - train_start
    
    result = {
        "dataset": dataset,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_basis": n_basis,
        "n_epochs": n_epochs,
        "n_train": n_train,
        "n_test": n_test,
        "preparation_time": prep_time,
        "training_time": train_time,
        "total_time": prep_time + train_time,
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "best_test_acc": max(history["test_acc"]),
        "history": history,
    }
    
    if verbose:
        print(f"Final: train_acc={result['final_train_acc']:.2%}, test_acc={result['final_test_acc']:.2%}")
        print(f"Training time: {train_time:.2f}s ({train_time/n_epochs*1000:.1f}ms/epoch)")
    
    return result


# =============================================================================
# Benchmark Functions
# =============================================================================


def bench_training(dataset: str, n_qubits: int, n_layers: int) -> dict | None:
    """Benchmark QTPU training."""
    print(f"\n{'='*60}")
    print(f"Training: {dataset}, qubits={n_qubits}, layers={n_layers}")
    print(f"{'='*60}")
    
    try:
        result = train_qtpu(dataset, n_qubits, n_layers, verbose=True)
        # Don't include full history in log (too large)
        result_log = {k: v for k, v in result.items() if k != "history"}
        return result_log
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Plotting
# =============================================================================


def plot_training_curves(results: list[dict], output_path: str = "plots/hybrid_ml/training_curves.pdf"):
    """Plot training curves for all configurations."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        dataset_results = [r for r in results if r["dataset"] == dataset]
        
        for r in dataset_results:
            label = f"q={r['n_qubits']}, l={r['n_layers']}"
            history = r["history"]
            ax.plot(history["epoch"], history["test_acc"], label=label, alpha=0.8)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{dataset.capitalize()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_final_accuracy(results: list[dict], output_path: str = "plots/hybrid_ml/final_accuracy.pdf"):
    """Bar plot of final test accuracy."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Group by dataset
    x_labels = []
    accuracies = []
    colors = []
    color_map = {"moons": "C0", "circles": "C1", "iris": "C2"}
    
    for dataset in DATASETS:
        dataset_results = [r for r in results if r["dataset"] == dataset]
        for r in dataset_results:
            x_labels.append(f"{dataset}\nq={r['n_qubits']},l={r['n_layers']}")
            accuracies.append(r["final_test_acc"])
            colors.append(color_map[dataset])
    
    x = np.arange(len(x_labels))
    bars = ax.bar(x, accuracies, color=colors, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("QTPU Training: Final Test Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.0%}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import sys
    
    usage = """
QTPU Training Demo

Usage: python training_demo.py <command>

Commands:
    run         Run full benchmark (all configs)
    quick       Quick demo (one config per dataset)
    plot        Plot results from logs

Configuration:
    Datasets: {DATASETS}
    Qubits: {NUM_QUBITS_LIST}
    Layers: {NUM_LAYERS_LIST}
    Epochs: {NUM_EPOCHS}
""".format(
        DATASETS=DATASETS,
        NUM_QUBITS_LIST=NUM_QUBITS_LIST,
        NUM_LAYERS_LIST=NUM_LAYERS_LIST,
        NUM_EPOCHS=NUM_EPOCHS,
    )
    
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "run":
        for dataset in DATASETS:
            for n_qubits in NUM_QUBITS_LIST:
                for n_layers in NUM_LAYERS_LIST:
                    config = {"dataset": dataset, "n_qubits": n_qubits, "n_layers": n_layers}
                    print(f"  Config: {config}")
                    result = bench_training(dataset, n_qubits, n_layers)
                    log_result("logs/hybrid_ml/qtpu_training_demo.jsonl", config, result)

    elif cmd == "quick":
        print("Quick training demo...")
        results = []
        for dataset in DATASETS:
            r = train_qtpu(dataset, n_qubits=4, n_layers=2, n_epochs=50, verbose=True)
            results.append(r)
        
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        for r in results:
            print(f"  {r['dataset']:10s}: test_acc={r['final_test_acc']:.2%}, time={r['total_time']:.2f}s")
        
        # Plot
        plot_training_curves(results, "plots/hybrid_ml/training_curves_quick.pdf")
        plot_final_accuracy(results, "plots/hybrid_ml/final_accuracy_quick.pdf")
        
    elif cmd == "plot":
        import json
        
        log_path = "logs/hybrid_ml/qtpu_training_demo.jsonl"
        results = []
        
        # Load results and re-run to get history (logs don't have it)
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                # Re-run to get history for plotting
                r = train_qtpu(
                    entry["dataset"],
                    entry["n_qubits"],
                    entry["n_layers"],
                    verbose=False
                )
                results.append(r)
        
        plot_training_curves(results)
        plot_final_accuracy(results)
        
    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
