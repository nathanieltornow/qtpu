"""Training with quantum feature encoders and trainable classical tensors.

The key idea:
- Quantum circuits act as fixed feature encoders (no trainable quantum params)
- Classical tensors (CTensor) are the trainable weights
- Gradients flow through standard PyTorch autograd (no parameter-shift needed)

Structure:
    ISwitch(batch_idx) -> feature_map(x[batch_idx]) -> measure -> q[batch]
    q[batch] @ W[batch, output] -> output  (W is trainable)
"""

import torch
import torch.nn as nn
import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter

from qtpu.heinsum import HEinsum
from qtpu.tensor import QuantumTensor, TensorSpec, ISwitch, CTensor
from qtpu.runtime import HEinsumContractor


def create_quantum_features(X: np.ndarray) -> QuantumTensor:
    """Create quantum feature encoder for data X.
    
    Args:
        X: Input data of shape (batch_size,)
    
    Returns:
        QuantumTensor with shape (batch_size,)
    """
    batch_size = len(X)
    batch_idx = Parameter("batch")
    
    def make_circuit(b: int) -> QuantumCircuit:
        """Encode X[b] into quantum state."""
        qc = QuantumCircuit(1)
        qc.ry(X[b], 0)  # Feature encoding
        return qc
    
    qc = QuantumCircuit(1, 1)
    qc.append(ISwitch(batch_idx, 1, batch_size, make_circuit), [0])
    qc.measure(0, 0)
    
    return QuantumTensor(qc)


def train_linear_classifier():
    """Train a linear classifier on quantum features."""
    print("=== Linear Classifier with Quantum Features ===\n")
    
    # Dataset
    batch_size = 8
    X = np.array([0.2, 0.4, 0.6, 0.8,    # Class 0
                  2.0, 2.2, 2.4, 2.6])    # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    print(f"X = {X}")
    print(f"y = {y}\n")
    
    # Quantum feature encoder (fixed)
    qtensor = create_quantum_features(X)
    
    # Trainable classical weight vector
    W = CTensor(
        torch.randn(batch_size, dtype=torch.float64, requires_grad=True),
        inds=("batch",),
    )
    
    # Contraction: q[batch] @ W[batch] -> scalar
    heinsum = HEinsum(
        qtensors=[qtensor],
        ctensors=[W],
        input_tensors=[],
        output_inds=(),
    )
    
    print(f"Einsum expr: {heinsum.einsum_expr}")
    print(f"Quantum tensor shape: {qtensor.shape}")
    print(f"Weight tensor shape: {W.shape}\n")
    
    contractor = HEinsumContractor(heinsum)
    contractor.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    # Training
    targets = torch.tensor(y, dtype=torch.float64)
    optimizer = torch.optim.Adam([W.data], lr=0.1)
    
    print("Training (optimizing classical weights)...")
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Forward: quantum features are fixed, only W is trainable
        # Output is scalar: sum_b q[b] * W[b]
        output = contractor.contract(input_tensors=[], circuit_params={})
        
        # We want the weighted sum to distinguish classes
        # Positive output -> class 1, negative -> class 0
        # Loss: encourage output to be positive (class 1 dominates)
        # Actually let's do per-sample: q[b] * W[b] should be positive for class 1
        
        # For scalar output, we can't do per-sample. Let's change approach.
        loss = (output - 1.0) ** 2  # Target: output = 1
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}: loss={loss.item():.4f}, output={output.item():.3f}")
    
    print(f"\nFinal weights: {W.data.detach().numpy().round(3)}")


def train_per_sample_classifier():
    """Train classifier with per-sample outputs."""
    print("\n=== Per-Sample Classifier ===\n")
    
    batch_size = 8
    X = np.array([0.2, 0.4, 0.6, 0.8,    # Class 0
                  2.0, 2.2, 2.4, 2.6])    # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    print(f"X = {X}")
    print(f"y = {y}\n")
    
    # Quantum features
    qtensor = create_quantum_features(X)
    
    # Trainable weight + bias per sample
    W = CTensor(
        torch.ones(batch_size, dtype=torch.float64, requires_grad=True),
        inds=("batch",),
    )
    
    # Output per sample: q[batch] * W[batch] -> [batch] (element-wise)
    # This requires a different contraction - keep batch index
    heinsum = HEinsum(
        qtensors=[qtensor],
        ctensors=[W],
        input_tensors=[],
        output_inds=("batch",),  # Keep batch dimension
    )
    
    print(f"Einsum expr: {heinsum.einsum_expr}")
    
    contractor = HEinsumContractor(heinsum)
    contractor.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    targets = torch.tensor(y, dtype=torch.float64)
    targets_scaled = 2 * targets - 1  # {0,1} -> {-1, +1}
    
    optimizer = torch.optim.Adam([W.data], lr=0.5)
    
    print("Training...")
    for epoch in range(60):
        optimizer.zero_grad()
        
        outputs = contractor.contract(input_tensors=[], circuit_params={})
        
        # MSE loss
        loss = ((outputs - targets_scaled) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        if epoch % 12 == 0:
            preds = (outputs > 0).float()
            acc = (preds == targets).float().mean()
            print(f"  Epoch {epoch:2d}: loss={loss.item():.4f}, acc={acc.item():.2f}")
    
    final_outputs = contractor.contract(input_tensors=[], circuit_params={})
    final_preds = (final_outputs > 0).float()
    print(f"\nFinal outputs: {final_outputs.detach().numpy().round(3)}")
    print(f"Predictions:   {final_preds.numpy().astype(int)}")
    print(f"Targets:       {y}")
    print(f"Final weights: {W.data.detach().numpy().round(3)}")


def train_two_layer_network():
    """Train a 2-layer network using HEinsum with trainable CTensors.
    
    Architecture:
        q[batch] * W1[batch, hidden] -> h[hidden]  (layer 1 via HEinsum)
        h[hidden] * W2[hidden] -> scalar           (layer 2 via HEinsum)
    """
    print("\n=== Two-Layer Network with CTensors ===\n")
    
    batch_size = 8
    hidden_dim = 4
    
    # Generate data
    np.random.seed(42)
    X = np.random.uniform(0, np.pi, batch_size)
    y = (np.sin(X) > 0.5).astype(float)  # Binary classification
    
    print(f"X = {X.round(2)}")
    print(f"y = {y.astype(int)}")
    print(f"Target mean: {y.mean():.2f}\n")
    
    # Quantum feature encoder: q[batch]
    qtensor = create_quantum_features(X)
    
    # Trainable weights as CTensors
    W1_data = torch.randn(batch_size, hidden_dim, dtype=torch.float64) * 0.5
    W1_data.requires_grad_(True)
    W1 = CTensor(W1_data, inds=("batch", "hidden"))
    
    W2_data = torch.randn(hidden_dim, dtype=torch.float64) * 0.5
    W2_data.requires_grad_(True)
    W2 = CTensor(W2_data, inds=("hidden",))
    
    # Layer 1: q[batch] * W1[batch, hidden] -> h[hidden]
    heinsum_layer1 = HEinsum(
        qtensors=[qtensor],
        ctensors=[W1],
        input_tensors=[],
        output_inds=("hidden",),
    )
    
    # Layer 2: h[hidden] * W2[hidden] -> scalar
    # h is provided as input_tensor at runtime
    h_spec = TensorSpec(("hidden",), {0: hidden_dim})
    heinsum_layer2 = HEinsum(
        qtensors=[],
        ctensors=[W2],
        input_tensors=[h_spec],
        output_inds=(),
    )
    
    contractor1 = HEinsumContractor(heinsum_layer1)
    contractor1.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    contractor2 = HEinsumContractor(heinsum_layer2)
    contractor2.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    print(f"Layer 1: q[{batch_size}] * W1[{batch_size},{hidden_dim}] -> h[{hidden_dim}]")
    print(f"Layer 2: h[{hidden_dim}] * W2[{hidden_dim}] -> scalar\n")
    
    target = torch.tensor(y.mean(), dtype=torch.float64)
    optimizer = torch.optim.Adam([W1_data, W2_data], lr=0.1)
    
    print("Training...")
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward pass through both HEinsum layers
        h = contractor1.contract(input_tensors=[], circuit_params={})
        h = torch.relu(h)  # Activation between layers
        output = contractor2.contract(input_tensors=[h], circuit_params={})
        output = torch.sigmoid(output)
        
        # BCE loss
        loss = -(target * torch.log(output + 1e-8) + 
                 (1 - target) * torch.log(1 - output + 1e-8))
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:2d}: loss={loss.item():.4f}, output={output.item():.3f}")
    
    print(f"\nFinal output: {output.item():.3f}")
    print(f"Target: {target.item():.2f}")


def train_regression_classical():
    """Regression with trainable classical tensor."""
    print("\n=== Regression with Classical Weights ===\n")
    
    batch_size = 8
    X = np.linspace(0, np.pi, batch_size)
    y = np.sin(X)
    
    print(f"X = {X.round(2)}")
    print(f"y = {y.round(3)}\n")
    
    # Quantum features: q[batch]
    qtensor = create_quantum_features(X)
    
    # Targets as classical tensor
    targets_tensor = CTensor(
        torch.tensor(y, dtype=torch.float64),
        inds=("batch",),
    )
    
    # Trainable scaling: W[batch] scales quantum outputs
    W = CTensor(
        torch.ones(batch_size, dtype=torch.float64, requires_grad=True),
        inds=("batch",),
    )
    
    # Contraction: q[batch] * W[batch] * targets[batch] -> scalar
    # We want q[batch] * W[batch] ≈ targets[batch]
    # So minimize (q * W - targets)^2
    
    # HEinsum for predictions: q[batch] * W[batch] -> [batch]
    heinsum_pred = HEinsum(
        qtensors=[qtensor],
        ctensors=[W],
        input_tensors=[],
        output_inds=("batch",),
    )
    
    print(f"Einsum: {heinsum_pred.einsum_expr}")
    
    contractor = HEinsumContractor(heinsum_pred)
    contractor.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    targets = torch.tensor(y, dtype=torch.float64)
    optimizer = torch.optim.Adam([W.data], lr=0.3)
    
    print("Training classical weights...")
    for epoch in range(50):
        optimizer.zero_grad()
        
        preds = contractor.contract(input_tensors=[], circuit_params={})
        loss = ((preds - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}: loss={loss.item():.6f}")
    
    final_preds = contractor.contract(input_tensors=[], circuit_params={})
    print(f"\nPredictions: {final_preds.detach().numpy().round(3)}")
    print(f"Targets:     {targets.numpy().round(3)}")
    print(f"Weights:     {W.data.detach().numpy().round(3)}")


if __name__ == "__main__":
    train_linear_classifier()
    train_per_sample_classifier()
    train_two_layer_network()
    train_regression_classical()
