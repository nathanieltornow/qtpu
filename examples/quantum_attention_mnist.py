"""
Quantum Multi-Head Attention for MNIST Classification
======================================================

A quantum attention architecture where:
  - Q (Query), K (Key), V (Value) are quantum feature extractors
  - Attention: softmax(Q @ K^T / sqrt(d)) @ V
  - Multiple heads capture different feature interactions

This demonstrates HEinsum's ability to:
  1. Express attention-like tensor contractions with quantum tensors
  2. Use more expressive variational quantum circuits
  3. Train end-to-end on real data
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit.circuit import QuantumCircuit, Parameter, ClassicalRegister
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qtpu.core import HEinsum, QuantumTensor, CTensor, ISwitch
from qtpu.runtime import HEinsumRuntime


# =============================================================================
# Data Loading
# =============================================================================

def load_mnist_binary(digit_a: int = 0, digit_b: int = 1, n_samples: int = 200):
    """Load MNIST and filter to two digits for binary classification."""
    print(f"Loading MNIST ({digit_a} vs {digit_b})...")
    
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Filter to two digits
    mask = (y == digit_a) | (y == digit_b)
    X, y = X[mask], y[mask]
    
    # Balance classes and limit samples
    n_per_class = n_samples // 2
    idx_a = np.where(y == digit_a)[0][:n_per_class]
    idx_b = np.where(y == digit_b)[0][:n_per_class]
    idx = np.concatenate([idx_a, idx_b])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    
    # Convert labels to 0/1
    y = (y == digit_b).astype(float)
    
    # Reshape to 28x28 and downsample to 8x8 (more resolution than 4x4)
    X = X.reshape(-1, 28, 28)
    X_down = np.zeros((len(X), 8, 8))
    for i in range(8):
        for j in range(8):
            # Average pool 3-4 pixel regions
            i_start, i_end = i * 28 // 8, (i + 1) * 28 // 8
            j_start, j_end = j * 28 // 8, (j + 1) * 28 // 8
            X_down[:, i, j] = X[:, i_start:i_end, j_start:j_end].mean(axis=(1, 2))
    
    # Flatten and normalize to [0, 2*pi] for angle encoding
    X_down = X_down.reshape(-1, 64)
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    X_down = scaler.fit_transform(X_down)
    
    return X_down, y


# =============================================================================
# Expressive Quantum Feature Circuits
# =============================================================================

def create_expressive_circuit(
    data: np.ndarray,
    role: str,  # "query", "key", or "value"
    num_qubits: int = 6,
    n_layers: int = 3,
    n_heads: int = 4,
) -> QuantumTensor:
    """Create an expressive quantum feature extractor.
    
    Architecture:
      - Hardware-efficient ansatz with data re-uploading
      - Multiple entanglement patterns
      - Role-specific rotations (Q, K, V have different structure)
    
    Output shape: [batch, head]
    """
    n_batch = len(data)
    n_features = data.shape[1]
    
    qc = QuantumCircuit(num_qubits)
    qc.add_register(ClassicalRegister(num_qubits))
    
    batch_param = Parameter("batch")
    head_param = Parameter(f"{role}_head")
    
    # Role-specific angle offsets for diversity
    role_offset = {"query": 0.0, "key": np.pi / 4, "value": np.pi / 2}[role]
    
    def make_data_circuit(idx: int) -> QuantumCircuit:
        """Expressive data encoding with re-uploading."""
        c = QuantumCircuit(num_qubits)
        x = data[idx]
        
        for layer in range(n_layers):
            # === Data encoding layer ===
            for q in range(num_qubits):
                # Use different features per qubit, cycling through
                feat_idx = (q + layer * num_qubits) % n_features
                angle = float(x[feat_idx]) + role_offset
                
                # Alternating rotation axes per layer
                if layer % 3 == 0:
                    c.ry(angle, q)
                elif layer % 3 == 1:
                    c.rz(angle, q)
                else:
                    c.rx(angle, q)
            
            # === Entanglement layer ===
            if layer % 2 == 0:
                # Linear entanglement
                for q in range(num_qubits - 1):
                    c.cx(q, q + 1)
            else:
                # Circular entanglement with SWAP-like
                for q in range(0, num_qubits - 1, 2):
                    c.cz(q, q + 1)
                for q in range(1, num_qubits - 1, 2):
                    c.cx(q, q + 1)
                # Close the ring
                c.cz(num_qubits - 1, 0)
            
            # === Feature interaction layer ===
            if layer < n_layers - 1:
                for q in range(num_qubits):
                    # Product features
                    f1_idx = (q * 2) % n_features
                    f2_idx = (q * 2 + 1) % n_features
                    prod_angle = float(x[f1_idx] * x[f2_idx]) / (2 * np.pi)
                    c.rz(prod_angle, q)
        
        return c
    
    def make_head_circuit(head_idx: int) -> QuantumCircuit:
        """Head-specific measurement basis rotation."""
        c = QuantumCircuit(num_qubits)
        
        # Each head measures in a different rotated basis
        base_angle = head_idx * np.pi / n_heads
        
        for q in range(num_qubits):
            # Varying angles per qubit within head
            angle = base_angle + q * np.pi / (2 * num_qubits)
            c.ry(angle, q)
            if head_idx % 2 == 1:
                c.rz(angle / 2, q)
        
        return c
    
    batch_iswitch = ISwitch(batch_param, num_qubits, n_batch, make_data_circuit)
    head_iswitch = ISwitch(head_param, num_qubits, n_heads, make_head_circuit)
    
    qc.append(batch_iswitch, range(num_qubits))
    qc.append(head_iswitch, range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    
    return QuantumTensor(qc)


# =============================================================================
# Quantum Multi-Head Attention
# =============================================================================

def create_attention_network(
    X: np.ndarray,
    n_heads: int = 4,
    num_qubits: int = 6,
    n_layers: int = 3,
) -> tuple[HEinsum, list[torch.Tensor]]:
    """Create quantum multi-head attention network.
    
    Architecture:
        Q[batch, head] ─┐
        K[batch, head] ─┼─ Attention weights W_attn[head, head] ─┐
        V[batch, head] ─┘                                        ├─ W_out[head] -> output[batch]
                                                                 
    The attention mechanism is:
        output[b] = sum_h V[b,h] * sum_h' softmax(Q[b,h'] * W_attn[h',h] * K[b,h])
    
    Simplified as tensor network:
        Q[b,h1] * W_qk[h1,h2] * K[b,h2] * W_v[h2,h3] * V[b,h3] * W_out[h3] -> [b]
    """
    print("Creating quantum Q/K/V tensors...")
    print(f"  Qubits: {num_qubits}, Layers: {n_layers}, Heads: {n_heads}")
    
    Q = create_expressive_circuit(X, "query", num_qubits, n_layers, n_heads)
    K = create_expressive_circuit(X, "key", num_qubits, n_layers, n_heads)
    V = create_expressive_circuit(X, "value", num_qubits, n_layers, n_heads)
    
    print(f"  Q: {Q.shape}, inds={Q.inds}")
    print(f"  K: {K.shape}, inds={K.inds}")
    print(f"  V: {V.shape}, inds={V.inds}")
    
    # Trainable attention weights
    # W_qk connects query and key heads (attention scores)
    W_qk_data = torch.randn(n_heads, n_heads, dtype=torch.float64) * 0.3
    W_qk_data.requires_grad_(True)
    W_qk = CTensor(W_qk_data, inds=("query_head", "key_head"))
    
    # W_kv connects key and value heads
    W_kv_data = torch.randn(n_heads, n_heads, dtype=torch.float64) * 0.3
    W_kv_data.requires_grad_(True)
    W_kv = CTensor(W_kv_data, inds=("key_head", "value_head"))
    
    # W_out projects value heads to scalar output
    W_out_data = torch.randn(n_heads, dtype=torch.float64) * 0.3
    W_out_data.requires_grad_(True)
    W_out = CTensor(W_out_data, inds=("value_head",))
    
    print(f"\nAttention weights:")
    print(f"  W_qk: {W_qk.shape}, inds={W_qk.inds}")
    print(f"  W_kv: {W_kv.shape}, inds={W_kv.inds}")
    print(f"  W_out: {W_out.shape}, inds={W_out.inds}")
    
    n_params = W_qk_data.numel() + W_kv_data.numel() + W_out_data.numel()
    print(f"  Total parameters: {n_params}")
    
    # HEinsum contraction:
    # Q[batch, query_head] * W_qk[query_head, key_head] * K[batch, key_head]
    # * W_kv[key_head, value_head] * V[batch, value_head] * W_out[value_head]
    # -> output[batch]
    heinsum = HEinsum(
        qtensors=[Q, K, V],
        ctensors=[W_qk, W_kv, W_out],
        input_tensors=[],
        output_inds=("batch",),
    )
    
    print(f"\nHEinsum expression: {heinsum.einsum_expr}")
    
    return heinsum, [W_qk_data, W_kv_data, W_out_data]


# =============================================================================
# Training
# =============================================================================

def train_attention_mnist():
    """Train quantum attention on MNIST."""
    print("=" * 70)
    print("QUANTUM MULTI-HEAD ATTENTION FOR MNIST")
    print("=" * 70)
    print()
    
    # Hyperparameters
    n_samples = 200
    n_heads = 8       # More heads for expressivity
    num_qubits = 4    # Fewer qubits (faster simulation)
    n_layers = 2      # Fewer layers (faster, less barren plateau)
    epochs = 80
    lr = 0.2
    
    # Load data
    X, y = load_mnist_binary(digit_a=0, digit_b=1, n_samples=n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"\nDataset: MNIST (0 vs 1)")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Features: {X.shape[1]} (8x8 downsampled)")
    print()
    
    # Create network
    print("-" * 70)
    print("BUILDING ATTENTION NETWORK")
    print("-" * 70)
    heinsum_train, weights = create_attention_network(
        X_train, n_heads=n_heads, num_qubits=num_qubits, n_layers=n_layers
    )
    
    # Learnable bias
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    
    # Create runtime
    print("\nPreparing HEinsum runtime...")
    runtime_train = HEinsumRuntime(heinsum_train, backend="cudaq", dtype=torch.float64)
    runtime_train.prepare(opt_kwargs={"max_repeats": 16, "progbar": False})
    
    if runtime_train.prep_timing:
        print(f"  Optimization time: {runtime_train.prep_timing.optimization_time*1000:.1f}ms")
    
    # Training
    print()
    print("-" * 70)
    print("TRAINING")
    print("-" * 70)
    
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    optimizer = torch.optim.Adam(weights + [bias], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_train_acc = 0.0
    total_quantum_time = 0.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        result, timing = runtime_train.execute()
        total_quantum_time += timing.quantum_eval_time
        
        # Logits with bias
        logits = result + bias
        probs = torch.sigmoid(logits)
        
        # Binary cross-entropy
        loss = torch.nn.functional.binary_cross_entropy(probs, y_train_t)
        
        # Backward and update
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track accuracy
        with torch.no_grad():
            preds = (probs > 0.5).float()
            acc = (preds == y_train_t).float().mean()
            if acc > best_train_acc:
                best_train_acc = acc
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, "
                  f"train_acc={acc:.2%}, "
                  f"quantum={timing.quantum_eval_time*1000:.0f}ms, "
                  f"lr={scheduler.get_last_lr()[0]:.4f}")
    
    # Summary
    print()
    print(f"Training Time: {total_quantum_time:.1f}s total, "
          f"{total_quantum_time/epochs*1000:.0f}ms/epoch")
    
    # Test evaluation
    print()
    print("-" * 70)
    print("EVALUATION")
    print("-" * 70)
    
    print("\nBuilding test network...")
    heinsum_test, _ = create_attention_network(
        X_test, n_heads=n_heads, num_qubits=num_qubits, n_layers=n_layers
    )
    
    # Copy trained weights
    for ctensor, trained_weight in zip(heinsum_test.classical_tensors, weights):
        ctensor._data = trained_weight
    
    runtime_test = HEinsumRuntime(heinsum_test, backend="cudaq", dtype=torch.float64)
    runtime_test.prepare(opt_kwargs={"max_repeats": 16, "progbar": False})
    
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    with torch.no_grad():
        # Train accuracy
        result_train, _ = runtime_train.execute()
        probs_train = torch.sigmoid(result_train + bias)
        train_acc = ((probs_train > 0.5).float() == y_train_t).float().mean()
        
        # Test accuracy
        result_test, _ = runtime_test.execute()
        probs_test = torch.sigmoid(result_test + bias)
        test_acc = ((probs_test > 0.5).float() == y_test_t).float().mean()
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy:  {test_acc:.2%}")
    print(f"  Learned bias:   {bias.item():.4f}")
    
    print(f"\nAttention weight analysis:")
    print(f"  W_qk (Q-K attention): norm={weights[0].norm().item():.4f}")
    print(f"  W_kv (K-V mixing):    norm={weights[1].norm().item():.4f}")
    print(f"  W_out (output proj):  norm={weights[2].norm().item():.4f}")
    
    # Show attention pattern
    print(f"\nLearned W_qk attention matrix:")
    with torch.no_grad():
        attn = torch.softmax(weights[0], dim=-1)
        print(attn.numpy().round(2))


if __name__ == "__main__":
    train_attention_mnist()
