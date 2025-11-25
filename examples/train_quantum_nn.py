"""Train a quantum circuit to learn a function using HTNLayer.

The ISwitch encodes a batch of training inputs (x values).
The circuit parameters (theta) are trained to approximate f(x) = sin(x).

This demonstrates:
- ISwitch as batch dimension for training data
- Training circuit params via parameter-shift gradients
- Learning a simple function with a variational quantum circuit
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit, Parameter

from qtpu.tensor import ISwitch
from qtpu.torch import HTNLayer, ExpvalTorchEvaluator


def create_data_encoding_circuit(x_values: np.ndarray, num_layers: int = 2) -> QuantumCircuit:
    """Create a variational circuit with data encoded via ISwitch.
    
    Architecture:
    - ISwitch encodes input x as RY(x) rotation (data encoding)
    - Variational layers with learnable RY, RZ rotations
    - Measure Z expectation value as output
    
    Args:
        x_values: Array of input values (batch)
        num_layers: Number of variational layers
    
    Returns:
        QuantumCircuit with ISwitch for batch and learnable params
    """
    batch_size = len(x_values)
    qc = QuantumCircuit(1)
    
    # ISwitch encodes different x values (batch dimension)
    batch_param = Parameter("b")  # batch index
    
    def data_encoder(idx: int) -> QuantumCircuit:
        """Encode x_values[idx] into qubit state."""
        enc = QuantumCircuit(1)
        enc.ry(x_values[idx], 0)  # Encode x as rotation angle
        return enc
    
    iswitch = ISwitch(batch_param, 1, batch_size, data_encoder)
    qc.append(iswitch, [0])
    
    # Variational layers (learnable)
    for layer in range(num_layers):
        qc.ry(Parameter(f"theta_{layer}"), 0)
        qc.rz(Parameter(f"phi_{layer}"), 0)
    
    qc.measure_all()
    return qc


def train_sin_function():
    """Train circuit to approximate sin(x)."""
    print("=" * 60)
    print("Training Quantum Circuit to Learn sin(x)")
    print("=" * 60)
    
    # Generate training data
    np.random.seed(42)
    n_train = 16
    x_train = np.linspace(0, 2 * np.pi, n_train)
    y_train = np.sin(x_train)  # Target: sin(x)
    
    print(f"Training data: {n_train} points in [0, 2π]")
    print(f"Target function: sin(x)")
    
    # Create circuit with batch encoding
    num_layers = 3
    qc = create_data_encoding_circuit(x_train, num_layers=num_layers)
    
    print(f"\nCircuit:")
    print(f"  - 1 qubit")
    print(f"  - {num_layers} variational layers")
    print(f"  - Learnable params: {[p.name for p in qc.parameters if p.name != 'b']}")
    
    # Create HTNLayer
    # einsum "b->b" means output has same batch dimension as input
    evaluator = ExpvalTorchEvaluator()
    
    layer = HTNLayer(
        einsum_expr="b->b",  # Keep batch dimension
        circuit_tensors=[qc],
        evaluator=evaluator,
        learnable_circuit_params=True,
    )
    
    layer.initialize_circuit_params(method="random", seed=123)
    
    print(f"\nHTNLayer:")
    print(f"  - Circuit params: {layer.num_circuit_params}")
    for name, param in layer.named_parameters():
        print(f"    {name}: {param.data.item():.4f}")
    
    # Target tensor
    y_target = torch.tensor(y_train, dtype=torch.float64)
    
    # Training
    optimizer = optim.Adam(layer.parameters(), lr=0.2)
    n_epochs = 100
    
    print(f"\nTraining for {n_epochs} epochs...")
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        y_pred = layer()  # Shape: (n_train,)
        loss = torch.mean((y_pred - y_target) ** 2)  # MSE
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss = {loss.item():.6f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    y_final = layer().detach().numpy()
    
    print("\nFinal parameters:")
    for name, param in layer.named_parameters():
        print(f"  {name}: {param.data.item():.4f}")
    
    print(f"\nFinal MSE: {losses[-1]:.6f}")
    
    # Test on new data
    print("\n--- Testing on new data ---")
    n_test = 32
    x_test = np.linspace(0, 2 * np.pi, n_test)
    y_test_true = np.sin(x_test)
    
    # Create new circuit with test data
    qc_test = create_data_encoding_circuit(x_test, num_layers=num_layers)
    
    layer_test = HTNLayer(
        einsum_expr="b->b",
        circuit_tensors=[qc_test],
        evaluator=evaluator,
        learnable_circuit_params=True,
    )
    
    # Copy learned parameters
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if hasattr(layer_test, name.replace(".", "_")) or name in dict(layer_test.named_parameters()):
                for test_name, test_param in layer_test.named_parameters():
                    if test_name == name:
                        test_param.data = param.data.clone()
    
    y_test_pred = layer_test().detach().numpy()
    test_mse = np.mean((y_test_pred - y_test_true) ** 2)
    print(f"Test MSE: {test_mse:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training curve
    axes[0].plot(losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True)
    
    # Predictions vs true
    axes[1].plot(x_test, y_test_true, "b-", label="sin(x)", linewidth=2)
    axes[1].plot(x_test, y_test_pred, "r--", label="Predicted", linewidth=2)
    axes[1].scatter(x_train, y_train, c="blue", s=30, alpha=0.5, label="Train data")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Function Approximation")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("sin_training.png", dpi=150)
    print(f"\nPlot saved to sin_training.png")
    plt.show()


def train_classification():
    """Train circuit for binary classification."""
    print("\n" + "=" * 60)
    print("Training Quantum Circuit for Binary Classification")
    print("=" * 60)
    
    # Generate classification data
    np.random.seed(42)
    n_samples = 20
    
    # Class 0: x in [0, π], Class 1: x in [π, 2π]
    x_class0 = np.random.uniform(0, np.pi, n_samples // 2)
    x_class1 = np.random.uniform(np.pi, 2 * np.pi, n_samples // 2)
    
    x_train = np.concatenate([x_class0, x_class1])
    y_train = np.concatenate([np.ones(n_samples // 2), -np.ones(n_samples // 2)])  # +1 / -1 labels
    
    # Shuffle
    perm = np.random.permutation(n_samples)
    x_train = x_train[perm]
    y_train = y_train[perm]
    
    print(f"Classification task:")
    print(f"  - Class +1: x ∈ [0, π]")
    print(f"  - Class -1: x ∈ [π, 2π]")
    print(f"  - {n_samples} training samples")
    
    # Create circuit
    qc = create_data_encoding_circuit(x_train, num_layers=2)
    evaluator = ExpvalTorchEvaluator()
    
    layer = HTNLayer(
        einsum_expr="b->b",
        circuit_tensors=[qc],
        evaluator=evaluator,
        learnable_circuit_params=True,
    )
    layer.initialize_circuit_params(method="random", seed=456)
    
    y_target = torch.tensor(y_train, dtype=torch.float64)
    
    # Training with hinge loss
    optimizer = optim.Adam(layer.parameters(), lr=0.3)
    
    print("\nTraining...")
    for epoch in range(50):
        optimizer.zero_grad()
        
        y_pred = layer()
        
        # Hinge loss: max(0, 1 - y * y_pred)
        loss = torch.mean(torch.clamp(1 - y_target * y_pred, min=0))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            accuracy = ((y_pred > 0) == (y_target > 0)).float().mean().item()
            print(f"  Epoch {epoch:3d}: loss = {loss.item():.4f}, accuracy = {accuracy:.2%}")
    
    # Final accuracy
    y_final = layer().detach()
    accuracy = ((y_final > 0) == (y_target > 0)).float().mean().item()
    print(f"\nFinal accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    train_sin_function()
    train_classification()
