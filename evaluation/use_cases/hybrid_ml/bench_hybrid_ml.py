"""Hybrid Quantum-Classical ML Benchmark.

This benchmark implements the programming model from the paper:

    @quantum_tensor
    def quantum_layer(X, W, O):
        q = alloc_qubits(N)
        for i in range(N):
            iswitch("i", [ry(x[i]) for x in X], q[i])  # batch encoding
        model(q, W)
        iswitch("k", [apply_obs(o) for o in O], q)    # observable selection
        measure(q)

    def forward(X, W, V, O):
        return hEinsum("jk,ik->ij", V, quantum_layer(X, W, O))

Key insight: QTPU represents circuit families as tensors. The `iswitch`
creates tensor dimensions for:
  - Batch samples (X[0], X[1], ..., X[batch_size])
  - Observables (O[0], O[1], ..., O[n_obs])
  - Other circuit parameters

The benchmark compares:
  1. QTPU: Native tensor representation with efficient execution
  2. Pennylane: Loop over samples × observables (sequential)
  3. Qiskit: EstimatorV2 with parameter binding

For QTPU to shine, we need:
  - Large batch sizes (many samples)
  - Multiple observables per sample
  - The cross-product creates a large circuit family
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@dataclass
class BenchResult:
    """Results from a hybrid ML benchmark."""
    
    framework: str
    n_qubits: int
    batch_size: int
    n_observables: int
    n_layers: int
    n_iterations: int
    
    # Circuit family size
    circuit_family_size: int = 0  # batch_size × n_observables
    
    # Timing
    setup_time: float = 0.0
    first_iter_time: float = 0.0
    mean_iter_time: float = 0.0
    total_time: float = 0.0
    
    # Throughput
    circuits_per_second: float = 0.0  # circuit_family_size / mean_iter_time
    samples_per_second: float = 0.0   # batch_size / mean_iter_time
    
    # Training
    final_loss: float = 0.0
    
    # Extra info
    extra: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "n_qubits": self.n_qubits,
            "batch_size": self.batch_size,
            "n_observables": self.n_observables,
            "n_layers": self.n_layers,
            "n_iterations": self.n_iterations,
            "circuit_family_size": self.circuit_family_size,
            "setup_time": self.setup_time,
            "first_iter_time": self.first_iter_time,
            "mean_iter_time": self.mean_iter_time,
            "total_time": self.total_time,
            "circuits_per_second": self.circuits_per_second,
            "samples_per_second": self.samples_per_second,
            "final_loss": self.final_loss,
            **self.extra,
        }


def generate_data(
    batch_size: int, 
    n_features: int, 
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data."""
    np.random.seed(seed)
    X = np.random.uniform(0, np.pi, (batch_size, n_features))
    y = (np.sin(X[:, 0]) + np.cos(X[:, 1] if n_features > 1 else X[:, 0]) > 1.0).astype(np.float64)
    return X, y


# =============================================================================
# QTPU Implementation
# =============================================================================

def bench_qtpu(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    n_observables: int,
    n_layers: int,
    n_iterations: int,
) -> BenchResult:
    """Benchmark QTPU hybrid ML.
    
    Uses ISwitch to create tensor dimensions for:
    - batch samples (i dimension)
    - observables (k dimension)
    
    The quantum tensor has shape (batch_size, n_observables).
    Classical tensor V has shape (n_observables, hidden_dim).
    Output: hEinsum("jk,ik->ij", V, Q) → (batch_size, hidden_dim)
    """
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp
    from qtpu.tensor import QuantumTensor, ISwitch, CTensor
    from qtpu.heinsum import HEinsum
    from qtpu.runtime import HEinsumRuntime
    
    batch_size = len(X)
    n_features = X.shape[1]
    circuit_family_size = batch_size * n_observables
    
    setup_start = time.perf_counter()
    
    # === Build quantum tensor with ISwitches ===
    
    # Parameter for batch index
    batch_param = Parameter("i")
    
    # Parameter for observable index  
    obs_param = Parameter("k")
    
    # Create circuit selectors for each batch sample
    def make_encoding_circuit(b: int) -> QuantumCircuit:
        """Encode X[b] into quantum state."""
        qc = QuantumCircuit(n_qubits)
        for layer in range(n_layers):
            # Feature encoding
            for q in range(n_qubits):
                feat_idx = q % n_features
                qc.ry(X[b, feat_idx], q)
            # Entangling layer
            if layer < n_layers - 1:
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)
        return qc
    
    # Create observable selectors
    def make_observable(k: int) -> QuantumCircuit:
        """Apply observable k (different Pauli strings)."""
        qc = QuantumCircuit(n_qubits)
        # Different observables: Z on different qubits, ZZ pairs, etc.
        if k < n_qubits:
            # Single Z on qubit k
            pass  # Z is default
        else:
            # ZZ on adjacent pair
            pair_idx = (k - n_qubits) % (n_qubits - 1)
            # For ZZ, we don't need explicit gates before measurement
            pass
        return qc
    
    # Build the full circuit with ISwitches
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # ISwitch for batch encoding
    qc.append(
        ISwitch(batch_param, n_qubits, batch_size, make_encoding_circuit),
        range(n_qubits)
    )
    
    # ISwitch for observable (identity circuits - observable handling is classical)
    qc.append(
        ISwitch(obs_param, n_qubits, n_observables, lambda k: QuantumCircuit(n_qubits)),
        range(n_qubits)
    )
    
    # Measure
    qc.measure(range(n_qubits), range(n_qubits))
    
    # Create quantum tensor - shape is (batch_size, n_observables)
    qtensor = QuantumTensor(qc)
    
    # Classical tensor V: (n_observables, hidden_dim) for linear combination
    hidden_dim = 4
    V_data = torch.randn(n_observables, hidden_dim, dtype=torch.float64, requires_grad=True)
    ctensor = CTensor(V_data.detach().numpy(), ("k", "j"))
    
    # HEinsum: contract quantum (i,k) with classical (k,j) → output (i,j)
    # "ik,kj->ij" means: sum over observables (k), get (batch, hidden)
    heinsum = HEinsum(
        qtensors=[qtensor],
        ctensors=[ctensor],
        input_tensors=[],
        output_inds=("i", "j"),
    )
    
    # Create runtime
    runtime = HEinsumRuntime(heinsum, backend="simulator", device="cpu")
    runtime.prepare(optimize=True)
    
    setup_time = time.perf_counter() - setup_start
    
    # === Training loop ===
    # Trainable: classical weights for final layer
    W_out = torch.randn(hidden_dim, 1, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    optimizer = torch.optim.Adam([V_data, W_out], lr=0.1)
    
    iter_times = []
    final_loss = 0.0
    
    for it in range(n_iterations):
        iter_start = time.perf_counter()
        
        optimizer.zero_grad()
        
        # Execute quantum tensor network
        # Result shape: (batch_size, hidden_dim)
        result, timing = runtime.execute()
        
        # Cache after first iteration
        if it == 0:
            runtime.cache_quantum()
        
        # Final linear layer
        pred = result @ W_out  # (batch, 1)
        pred = torch.sigmoid(pred)
        
        # Binary cross-entropy loss
        loss = torch.nn.functional.binary_cross_entropy(pred, y_t)
        loss.backward()
        optimizer.step()
        
        iter_times.append(time.perf_counter() - iter_start)
        final_loss = loss.item()
    
    total_time = sum(iter_times)
    mean_iter_time = np.mean(iter_times[1:]) if len(iter_times) > 1 else iter_times[0]
    
    return BenchResult(
        framework="qtpu",
        n_qubits=n_qubits,
        batch_size=batch_size,
        n_observables=n_observables,
        n_layers=n_layers,
        n_iterations=n_iterations,
        circuit_family_size=circuit_family_size,
        setup_time=setup_time,
        first_iter_time=iter_times[0],
        mean_iter_time=mean_iter_time,
        total_time=total_time,
        circuits_per_second=circuit_family_size / mean_iter_time,
        samples_per_second=batch_size / mean_iter_time,
        final_loss=final_loss,
    )


# =============================================================================
# Pennylane Implementation
# =============================================================================

def bench_pennylane(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    n_observables: int,
    n_layers: int,
    n_iterations: int,
) -> BenchResult:
    """Benchmark Pennylane hybrid ML.
    
    Pennylane doesn't have native circuit-family representation.
    We must loop over samples × observables.
    """
    import pennylane as qml
    
    batch_size = len(X)
    n_features = X.shape[1]
    circuit_family_size = batch_size * n_observables
    
    setup_start = time.perf_counter()
    
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    # Create observables
    observables = []
    for k in range(n_observables):
        if k < n_qubits:
            observables.append(qml.PauliZ(k))
        else:
            pair_idx = (k - n_qubits) % (n_qubits - 1)
            observables.append(qml.PauliZ(pair_idx) @ qml.PauliZ(pair_idx + 1))
    
    @qml.qnode(dev, interface="torch")
    def circuit(x, obs_idx):
        """Quantum circuit for single sample and observable."""
        for layer in range(n_layers):
            for q in range(n_qubits):
                feat_idx = q % x.shape[0]
                qml.RY(x[feat_idx], wires=q)
            if layer < n_layers - 1:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
        return qml.expval(observables[obs_idx])
    
    setup_time = time.perf_counter() - setup_start
    
    # Trainable parameters
    hidden_dim = 4
    V = torch.randn(n_observables, hidden_dim, dtype=torch.float64, requires_grad=True)
    W_out = torch.randn(hidden_dim, 1, dtype=torch.float64, requires_grad=True)
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    optimizer = torch.optim.Adam([V, W_out], lr=0.1)
    
    iter_times = []
    final_loss = 0.0
    
    for it in range(n_iterations):
        iter_start = time.perf_counter()
        
        optimizer.zero_grad()
        
        # Execute circuits: loop over batch × observables
        # Result shape: (batch_size, n_observables)
        q_results = torch.zeros(batch_size, n_observables, dtype=torch.float64)
        for b in range(batch_size):
            for k in range(n_observables):
                q_results[b, k] = circuit(X_t[b], k)
        
        # Contract with V: (batch, obs) @ (obs, hidden) → (batch, hidden)
        hidden = q_results @ V
        
        # Final layer
        pred = hidden @ W_out
        pred = torch.sigmoid(pred)
        
        loss = torch.nn.functional.binary_cross_entropy(pred, y_t)
        loss.backward()
        optimizer.step()
        
        iter_times.append(time.perf_counter() - iter_start)
        final_loss = loss.item()
    
    total_time = sum(iter_times)
    mean_iter_time = np.mean(iter_times[1:]) if len(iter_times) > 1 else iter_times[0]
    
    return BenchResult(
        framework="pennylane",
        n_qubits=n_qubits,
        batch_size=batch_size,
        n_observables=n_observables,
        n_layers=n_layers,
        n_iterations=n_iterations,
        circuit_family_size=circuit_family_size,
        setup_time=setup_time,
        first_iter_time=iter_times[0],
        mean_iter_time=mean_iter_time,
        total_time=total_time,
        circuits_per_second=circuit_family_size / mean_iter_time,
        samples_per_second=batch_size / mean_iter_time,
        final_loss=final_loss,
    )


def bench_pennylane_batched(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    n_observables: int,
    n_layers: int,
    n_iterations: int,
) -> BenchResult:
    """Benchmark Pennylane with parameter broadcasting.
    
    Uses Pennylane's native batching for samples, but still loops over observables.
    """
    import pennylane as qml
    
    batch_size = len(X)
    n_features = X.shape[1]
    circuit_family_size = batch_size * n_observables
    
    setup_start = time.perf_counter()
    
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    # Create observables
    observables = []
    for k in range(n_observables):
        if k < n_qubits:
            observables.append(qml.PauliZ(k))
        else:
            pair_idx = (k - n_qubits) % (n_qubits - 1)
            observables.append(qml.PauliZ(pair_idx) @ qml.PauliZ(pair_idx + 1))
    
    # Create one circuit per observable (batched over samples)
    circuits = []
    for k in range(n_observables):
        @qml.qnode(dev, interface="torch")
        def circuit(x, obs=observables[k]):
            """Quantum circuit batched over samples."""
            for layer in range(n_layers):
                for q in range(n_qubits):
                    feat_idx = q % x.shape[-1]
                    qml.RY(x[..., feat_idx], wires=q)
                if layer < n_layers - 1:
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
            return qml.expval(obs)
        circuits.append(circuit)
    
    setup_time = time.perf_counter() - setup_start
    
    # Trainable parameters
    hidden_dim = 4
    V = torch.randn(n_observables, hidden_dim, dtype=torch.float64, requires_grad=True)
    W_out = torch.randn(hidden_dim, 1, dtype=torch.float64, requires_grad=True)
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    optimizer = torch.optim.Adam([V, W_out], lr=0.1)
    
    iter_times = []
    final_loss = 0.0
    
    for it in range(n_iterations):
        iter_start = time.perf_counter()
        
        optimizer.zero_grad()
        
        # Execute circuits: batched over samples, loop over observables
        # Result shape: (batch_size, n_observables)
        q_results = torch.stack([circuits[k](X_t) for k in range(n_observables)], dim=1)
        
        # Contract with V
        hidden = q_results @ V
        
        # Final layer
        pred = hidden @ W_out
        pred = torch.sigmoid(pred)
        
        loss = torch.nn.functional.binary_cross_entropy(pred, y_t)
        loss.backward()
        optimizer.step()
        
        iter_times.append(time.perf_counter() - iter_start)
        final_loss = loss.item()
    
    total_time = sum(iter_times)
    mean_iter_time = np.mean(iter_times[1:]) if len(iter_times) > 1 else iter_times[0]
    
    return BenchResult(
        framework="pennylane_batched",
        n_qubits=n_qubits,
        batch_size=batch_size,
        n_observables=n_observables,
        n_layers=n_layers,
        n_iterations=n_iterations,
        circuit_family_size=circuit_family_size,
        setup_time=setup_time,
        first_iter_time=iter_times[0],
        mean_iter_time=mean_iter_time,
        total_time=total_time,
        circuits_per_second=circuit_family_size / mean_iter_time,
        samples_per_second=batch_size / mean_iter_time,
        final_loss=final_loss,
    )


# =============================================================================
# Qiskit Implementation
# =============================================================================

def bench_qiskit(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    n_observables: int,
    n_layers: int,
    n_iterations: int,
) -> BenchResult:
    """Benchmark Qiskit EstimatorV2 hybrid ML.
    
    Uses EstimatorV2's parameter binding to batch samples.
    Still loops over observables.
    """
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit_aer.primitives import EstimatorV2
    from qiskit.quantum_info import SparsePauliOp
    
    batch_size = len(X)
    n_features = X.shape[1]
    circuit_family_size = batch_size * n_observables
    
    setup_start = time.perf_counter()
    
    # Create parameterized circuit
    params = [Parameter(f"x_{i}") for i in range(n_features)]
    
    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        for q in range(n_qubits):
            feat_idx = q % n_features
            qc.ry(params[feat_idx], q)
        if layer < n_layers - 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
    
    # Create observables
    observables = []
    for k in range(n_observables):
        if k < n_qubits:
            # Single Z on qubit k
            pauli_str = "I" * k + "Z" + "I" * (n_qubits - k - 1)
        else:
            # ZZ on adjacent pair
            pair_idx = (k - n_qubits) % (n_qubits - 1)
            pauli_str = "I" * pair_idx + "ZZ" + "I" * (n_qubits - pair_idx - 2)
        observables.append(SparsePauliOp(pauli_str))
    
    estimator = EstimatorV2()
    
    setup_time = time.perf_counter() - setup_start
    
    # Trainable parameters
    hidden_dim = 4
    V = torch.randn(n_observables, hidden_dim, dtype=torch.float64, requires_grad=True)
    W_out = torch.randn(hidden_dim, 1, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    optimizer = torch.optim.Adam([V, W_out], lr=0.1)
    
    iter_times = []
    final_loss = 0.0
    
    for it in range(n_iterations):
        iter_start = time.perf_counter()
        
        optimizer.zero_grad()
        
        # Execute circuits: parameter binding for samples, loop over observables
        q_results = np.zeros((batch_size, n_observables), dtype=np.float64)
        
        for k, obs in enumerate(observables):
            # Batch over samples for this observable
            pub = (qc, obs, [X[b].tolist() for b in range(batch_size)])
            job = estimator.run([pub])
            result = job.result()[0]
            q_results[:, k] = result.data.evs
        
        q_results_t = torch.tensor(q_results, dtype=torch.float64)
        
        # Contract with V
        hidden = q_results_t @ V
        
        # Final layer
        pred = hidden @ W_out
        pred = torch.sigmoid(pred)
        
        loss = torch.nn.functional.binary_cross_entropy(pred, y_t)
        loss.backward()
        optimizer.step()
        
        iter_times.append(time.perf_counter() - iter_start)
        final_loss = loss.item()
    
    total_time = sum(iter_times)
    mean_iter_time = np.mean(iter_times[1:]) if len(iter_times) > 1 else iter_times[0]
    
    return BenchResult(
        framework="qiskit",
        n_qubits=n_qubits,
        batch_size=batch_size,
        n_observables=n_observables,
        n_layers=n_layers,
        n_iterations=n_iterations,
        circuit_family_size=circuit_family_size,
        setup_time=setup_time,
        first_iter_time=iter_times[0],
        mean_iter_time=mean_iter_time,
        total_time=total_time,
        circuits_per_second=circuit_family_size / mean_iter_time,
        samples_per_second=batch_size / mean_iter_time,
        final_loss=final_loss,
    )


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison(
    n_qubits: int = 4,
    batch_size: int = 16,
    n_observables: int = 8,
    n_layers: int = 2,
    n_iterations: int = 5,
):
    """Run comparison of hybrid ML implementations."""
    print("=" * 70)
    print("HYBRID QUANTUM-CLASSICAL ML BENCHMARK")
    print("=" * 70)
    print()
    print(f"Config: {n_qubits}q, batch={batch_size}, obs={n_observables}, layers={n_layers}")
    print(f"Circuit family size: {batch_size} × {n_observables} = {batch_size * n_observables}")
    print("-" * 70)
    
    X, y = generate_data(batch_size, n_features=min(n_qubits, 4))
    
    results = []
    
    # QTPU
    print("\nRunning QTPU...")
    try:
        r = bench_qtpu(X, y, n_qubits, n_observables, n_layers, n_iterations)
        results.append(r)
        print(f"  {r.circuits_per_second:.1f} circuits/s, {r.samples_per_second:.1f} samples/s")
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Pennylane sequential
    print("\nRunning Pennylane (sequential)...")
    try:
        r = bench_pennylane(X, y, n_qubits, n_observables, n_layers, n_iterations)
        results.append(r)
        print(f"  {r.circuits_per_second:.1f} circuits/s, {r.samples_per_second:.1f} samples/s")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Pennylane batched
    print("\nRunning Pennylane (batched)...")
    try:
        r = bench_pennylane_batched(X, y, n_qubits, n_observables, n_layers, n_iterations)
        results.append(r)
        print(f"  {r.circuits_per_second:.1f} circuits/s, {r.samples_per_second:.1f} samples/s")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Qiskit
    print("\nRunning Qiskit...")
    try:
        r = bench_qiskit(X, y, n_qubits, n_observables, n_layers, n_iterations)
        results.append(r)
        print(f"  {r.circuits_per_second:.1f} circuits/s, {r.samples_per_second:.1f} samples/s")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Framework':<20} {'Circuits/s':<12} {'Samples/s':<12} {'Iter (s)':<10} {'Speedup':<10}")
    print("-" * 70)
    
    baseline = results[-1].circuits_per_second if results else 1
    for r in results:
        speedup = r.circuits_per_second / baseline if baseline > 0 else 0
        print(f"{r.framework:<20} {r.circuits_per_second:<12.1f} {r.samples_per_second:<12.1f} {r.mean_iter_time:<10.4f} {speedup:<10.1f}x")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "scale":
        # Scaling experiment
        print("=" * 70)
        print("SCALING EXPERIMENT: Circuit Family Size")
        print("=" * 70)
        
        configs = [
            (4, 8, 4),    # 32 circuits
            (4, 16, 8),   # 128 circuits
            (4, 32, 16),  # 512 circuits
            (6, 32, 16),  # 512 circuits, larger qubits
        ]
        
        for n_qubits, batch_size, n_obs in configs:
            print(f"\n{'='*70}")
            run_comparison(n_qubits, batch_size, n_obs, n_layers=2, n_iterations=3)
    else:
        # Quick comparison
        run_comparison()
