"""End-to-end training with ISwitch as batch dimension.

The key idea:
- ISwitch selects different feature-map circuits for each data point x in X
- A trainable quantum layer follows (shared across batch)
- Classical contraction sums over batch to compute loss

Structure:
    ISwitch(batch_idx) -> feature_map(x[batch_idx]) -> trainable_layer(theta) -> measure
"""

import torch
import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter

from qtpu.heinsum import HEinsum
from qtpu.tensor import QuantumTensor, ISwitch
from qtpu.runtime import HEinsumContractor


def create_data_encoding_layer(X: np.ndarray) -> HEinsum:
    """Create a quantum layer where ISwitch encodes different data points.
    
    For each data point x in X:
        1. Feature map: encode x into quantum state via Ry(x)
        2. Trainable layer: apply Ry(theta) 
        3. Measure
    
    Args:
        X: Input data array of shape (batch_size,) - one feature per sample
    
    Returns:
        HEinsum with batch dimension from ISwitch
    """
    batch_size = len(X)
    
    # Parameters
    batch_idx = Parameter("batch")
    theta = Parameter("theta")  # Trainable
    
    def make_circuit(b: int) -> QuantumCircuit:
        """Circuit for batch sample b: feature_map(X[b]) + trainable(theta)."""
        qc = QuantumCircuit(1)
        # Feature map: encode data point X[b]
        qc.ry(X[b], 0)
        # Trainable layer
        qc.ry(theta, 0)
        return qc
    
    # Main circuit with ISwitch over batch
    qc = QuantumCircuit(1, 1)
    qc.append(ISwitch(batch_idx, 1, batch_size, make_circuit), [0])
    qc.measure(0, 0)
    
    qtensor = QuantumTensor(qc)
    
    # Output: q[batch] with output_inds=("batch",) gives full batch output
    heinsum = HEinsum(
        qtensors=[qtensor],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch",),
    )
    
    return heinsum


def train_regression():
    """Train to predict target values from input features."""
    print("=== Regression Training ===\n")
    
    # Generate simple dataset: y = sin(x)
    np.random.seed(42)
    batch_size = 8
    X = np.linspace(0, np.pi, batch_size)
    y = np.sin(X)
    
    print(f"Data: X = {X.round(2)}")
    print(f"      y = {y.round(2)}")
    print()
    
    # Create HEinsum with data encoding
    heinsum = create_data_encoding_layer(X)
    
    print(f"Einsum expr: {heinsum.einsum_expr}")
    print(f"Quantum tensor shape: {heinsum.quantum_tensors[0].shape}")
    print(f"Quantum tensor inds: {heinsum.quantum_tensors[0].inds}")
    print()
    
    # Prepare contractor
    contractor = HEinsumContractor(heinsum)
    contractor.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    # Target as torch tensor
    targets = torch.tensor(y, dtype=torch.float64)
    
    # Trainable parameter
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=0.2)
    
    print("Training...")
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Forward: get predictions for all batch samples
        preds = contractor.contract(input_tensors=[], circuit_params={"theta": theta})
        
        # MSE loss
        loss = ((preds - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}: loss={loss.item():.6f}, theta={theta.item():.4f}")
    
    # Final predictions
    final_preds = contractor.contract(input_tensors=[], circuit_params={"theta": theta})
    print(f"\nFinal predictions: {final_preds.detach().numpy().round(3)}")
    print(f"Targets:           {targets.numpy().round(3)}")
    print(f"Final theta: {theta.item():.4f}")


def train_classification():
    """Train binary classification."""
    print("\n=== Binary Classification ===\n")
    
    # Simple dataset: classify based on angle
    batch_size = 8
    X = np.array([0.1, 0.3, 0.5, 0.7,   # Class 0: small angles
                  2.0, 2.2, 2.5, 2.8])   # Class 1: large angles
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    print(f"X = {X}")
    print(f"y = {y}")
    print()
    
    # Create HEinsum
    heinsum = create_data_encoding_layer(X)
    contractor = HEinsumContractor(heinsum)
    contractor.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    targets = torch.tensor(y, dtype=torch.float64)
    
    # Trainable parameter
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=0.3)
    
    print("Training...")
    for epoch in range(40):
        optimizer.zero_grad()
        
        # Get raw outputs (expectation values in [-1, 1])
        outputs = contractor.contract(input_tensors=[], circuit_params={"theta": theta})
        
        # MSE loss: treat y=0 as output=-1, y=1 as output=+1
        targets_scaled = 2 * targets - 1  # Map {0,1} -> {-1, +1}
        loss = ((outputs - targets_scaled) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        if epoch % 8 == 0:
            preds = (outputs > 0).float()
            acc = (preds == targets).float().mean()
            print(f"  Epoch {epoch:2d}: loss={loss.item():.4f}, acc={acc.item():.2f}, theta={theta.item():.3f}")
    
    # Final accuracy
    final_outputs = contractor.contract(input_tensors=[], circuit_params={"theta": theta})
    final_preds = (final_outputs > 0).float()
    final_acc = (final_preds == targets).float().mean()
    print(f"\nFinal accuracy: {final_acc.item():.2f}")
    print(f"Predictions: {final_preds.numpy().astype(int)}")
    print(f"Targets:     {targets.numpy().astype(int)}")


def train_multi_param():
    """Train with multiple trainable parameters."""
    print("\n=== Multi-Parameter Training ===\n")
    
    batch_size = 6
    X = np.linspace(0, 2, batch_size)
    y = np.cos(X)  # Target: cos(x)
    
    # Parameters
    batch_idx = Parameter("batch")
    theta1 = Parameter("theta1")
    theta2 = Parameter("theta2")
    
    def make_circuit(b: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(X[b], 0)       # Feature map
        qc.ry(theta1, 0)     # Trainable layer 1
        qc.rz(theta2, 0)     # Trainable layer 2
        return qc
    
    qc = QuantumCircuit(1, 1)
    qc.append(ISwitch(batch_idx, 1, batch_size, make_circuit), [0])
    qc.measure(0, 0)
    
    heinsum = HEinsum(
        qtensors=[QuantumTensor(qc)],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch",),
    )
    
    contractor = HEinsumContractor(heinsum)
    contractor.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    targets = torch.tensor(y, dtype=torch.float64)
    
    # Multiple trainable parameters
    theta1_val = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    theta2_val = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta1_val, theta2_val], lr=0.2)
    
    print(f"Targets: {y.round(3)}")
    print("\nTraining with 2 parameters...")
    
    for epoch in range(60):
        optimizer.zero_grad()
        
        preds = contractor.contract(
            input_tensors=[],
            circuit_params={"theta1": theta1_val, "theta2": theta2_val}
        )
        
        loss = ((preds - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        if epoch % 12 == 0:
            print(f"  Epoch {epoch:2d}: loss={loss.item():.6f}")
    
    final_preds = contractor.contract(
        input_tensors=[],
        circuit_params={"theta1": theta1_val, "theta2": theta2_val}
    )
    print(f"\nFinal predictions: {final_preds.detach().numpy().round(3)}")
    print(f"Targets:           {targets.numpy().round(3)}")
    print(f"theta1={theta1_val.item():.3f}, theta2={theta2_val.item():.3f}")


if __name__ == "__main__":
    train_regression()
    train_classification()
    train_multi_param()
