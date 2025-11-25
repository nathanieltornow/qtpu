"""Example: Using HTNLayer as a differentiable PyTorch layer.

This example demonstrates how to:
1. Create parameterized quantum circuits with ISwitches
2. Define contraction via einsum expression
3. Train circuit parameters using gradient descent (parameter-shift rule)

HTNLayer API:
    __init__(einsum_expr, circuit_tensors, evaluator, classical_tensors=None, learnable_circuit_params=True)
    
    - einsum_expr: "ij,jk,kl->il" style contraction
    - circuit_tensors: first m tensors in einsum (quantum)
    - classical_tensors: next k tensors (learnable nn.Parameters)
    - remaining n tensors provided at forward time
    - learnable_circuit_params: if True, circuit params are nn.Parameters
    
    forward(*input_tensors, circuit_params=None)
"""

import torch
import torch.optim as optim
import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter

from qtpu.tensor import ISwitch
from qtpu.torch import (
    HTNLayer,
    ParameterizedCircuitTensor,
    ExpvalTorchEvaluator,
)


def create_simple_circuit(
    num_qubits: int = 2,
    batch_size: int = 4,
) -> QuantumCircuit:
    """Create a simple variational circuit with batch ISwitch.
    
    Args:
        num_qubits: Number of qubits.
        batch_size: Batch dimension size (ISwitch).
    
    Returns:
        QuantumCircuit with one ISwitch (batch) and learnable theta.
    """
    qc = QuantumCircuit(num_qubits)
    
    # ISwitch for batch dimension
    batch_param = Parameter("i")  # Will be index "i" in einsum
    
    def input_selector(idx: int) -> QuantumCircuit:
        input_qc = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            angle = 2 * np.pi * (idx + q) / batch_size
            input_qc.ry(angle, q)
        return input_qc
    
    iswitch = ISwitch(batch_param, num_qubits, batch_size, input_selector)
    qc.append(iswitch, range(num_qubits))
    
    # Learnable rotation
    theta = Parameter("theta")
    for q in range(num_qubits):
        qc.ry(theta, q)
    
    qc.measure_all()
    return qc


def example_basic():
    """Basic HTNLayer usage."""
    print("=" * 60)
    print("Example: Basic HTNLayer")
    print("=" * 60)
    
    batch_size = 4
    
    # Create circuit with ISwitch index "i" and learnable "theta"
    qc = create_simple_circuit(num_qubits=2, batch_size=batch_size)
    
    print(f"Circuit parameters: {[p.name for p in qc.parameters]}")
    print(f"  - 'i': ISwitch (batch dimension, size {batch_size})")
    print(f"  - 'theta': Learnable parameter")
    
    # Wrap as ParameterizedCircuitTensor
    pct = ParameterizedCircuitTensor(qc)
    print(f"\nCircuitTensor shape: {pct.shape}")
    print(f"CircuitTensor indices: {pct.inds}")
    print(f"Learnable params: {pct.learnable_params}")
    
    # Create evaluator
    evaluator = ExpvalTorchEvaluator()
    
    # HTNLayer: one circuit tensor "i", contracted with weight vector "i"
    # einsum: "i,i->" means sum over batch with weights
    layer = HTNLayer(
        einsum_expr="i,i->",
        circuit_tensors=[qc],
        evaluator=evaluator,
        classical_tensors=None,  # weights passed at forward
        learnable_circuit_params=True,  # theta is learned
    )
    
    print(f"\nHTNLayer created:")
    print(f"  - Circuit tensors: {layer.num_circuit_tensors}")
    print(f"  - Circuit params: {layer.num_circuit_params}")
    print(f"  - Learnable classical: {layer.num_learnable_classical}")
    print(f"  - Input tensors at forward: {layer.num_input_tensors}")
    
    # Forward pass (no circuit_params needed since they're learned)
    weights = torch.ones(batch_size, dtype=torch.float64)
    result = layer(weights)
    print(f"\nForward result: {result.item():.4f}")


def example_training():
    """Train circuit parameters using gradient descent."""
    print("\n" + "=" * 60)
    print("Example: Training Circuit Parameters")
    print("=" * 60)
    
    batch_size = 4
    qc = create_simple_circuit(num_qubits=1, batch_size=batch_size)
    
    evaluator = ExpvalTorchEvaluator()
    
    # Create layer with learnable circuit params and classical weights
    initial_weights = torch.ones(batch_size, dtype=torch.float64)
    
    layer = HTNLayer(
        einsum_expr="i,i->",
        circuit_tensors=[qc],
        evaluator=evaluator,
        classical_tensors=[initial_weights],  # becomes nn.Parameter
        learnable_circuit_params=True,
    )
    
    # Initialize circuit params
    layer.initialize_circuit_params(method="random", seed=42)
    
    print(f"Learnable parameters:")
    for name, param in layer.named_parameters():
        print(f"  {name}: {param.data.item() if param.dim() == 0 else param.shape}")
    
    # Training loop - maximize output
    optimizer = optim.Adam(layer.parameters(), lr=0.1)
    
    print("\nTraining (maximizing output)...")
    for epoch in range(10):
        optimizer.zero_grad()
        
        output = layer()  # No input tensors needed
        loss = -output  # Maximize = minimize negative
        
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            theta = layer.get_circuit_param(0, "theta")
            print(f"  Epoch {epoch}: output = {output.item():.4f}, theta = {theta.item():.4f}")
    
    print("\nFinal parameters:")
    for name, param in layer.named_parameters():
        val = param.data.item() if param.dim() == 0 else param.data.numpy()
        print(f"  {name}: {val}")


def example_target_state():
    """Train to match a target output."""
    print("\n" + "=" * 60)
    print("Example: Training to Match Target")
    print("=" * 60)
    
    batch_size = 4
    
    # Simple circuit: just RY rotation
    qc = QuantumCircuit(1)
    i_param = Parameter("i")
    
    def selector(idx):
        c = QuantumCircuit(1)
        # Different initial state for each batch element
        c.ry(idx * np.pi / 4, 0)
        return c
    
    qc.append(ISwitch(i_param, 1, batch_size, selector), [0])
    qc.ry(Parameter("theta"), 0)
    qc.measure_all()
    
    evaluator = ExpvalTorchEvaluator()
    
    # Target: specific expectation values for each batch element
    target = torch.tensor([0.5, 0.0, -0.5, -0.8], dtype=torch.float64)
    
    layer = HTNLayer(
        einsum_expr="i->i",  # Output has batch dimension
        circuit_tensors=[qc],
        evaluator=evaluator,
        learnable_circuit_params=True,
    )
    
    layer.initialize_circuit_params(method="zeros")
    
    optimizer = optim.Adam(layer.parameters(), lr=0.3)
    
    print(f"Target: {target.numpy()}")
    print("\nTraining...")
    
    for epoch in range(20):
        optimizer.zero_grad()
        
        output = layer()
        loss = torch.mean((output - target) ** 2)  # MSE loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss = {loss.item():.4f}")
            print(f"    output = {output.detach().numpy()}")
    
    print(f"\nFinal output: {layer().detach().numpy()}")
    print(f"Target:       {target.numpy()}")


def example_two_circuits():
    """HTNLayer with two circuit tensors."""
    print("\n" + "=" * 60)
    print("Example: Two Circuit Tensors")
    print("=" * 60)
    
    batch_size = 3
    
    # Circuit 1: index "i", learnable "theta"
    qc1 = QuantumCircuit(1)
    i_param = Parameter("i")
    
    def selector1(idx):
        c = QuantumCircuit(1)
        c.rx(idx * np.pi / batch_size, 0)
        return c
    
    qc1.append(ISwitch(i_param, 1, batch_size, selector1), [0])
    qc1.ry(Parameter("theta"), 0)
    qc1.measure_all()
    
    # Circuit 2: index "j", learnable "phi"  
    qc2 = QuantumCircuit(1)
    j_param = Parameter("j")
    
    def selector2(idx):
        c = QuantumCircuit(1)
        c.rz(idx * np.pi / batch_size, 0)
        return c
    
    qc2.append(ISwitch(j_param, 1, batch_size, selector2), [0])
    qc2.rz(Parameter("phi"), 0)
    qc2.measure_all()
    
    evaluator = ExpvalTorchEvaluator()
    
    # Einsum: contract two circuits with a matrix
    # "i,j,ij->" = sum over i,j with interaction matrix
    interaction = torch.randn(batch_size, batch_size, dtype=torch.float64)
    
    layer = HTNLayer(
        einsum_expr="i,j,ij->",
        circuit_tensors=[qc1, qc2],
        evaluator=evaluator,
        classical_tensors=[interaction],  # learnable
        learnable_circuit_params=True,
    )
    
    print(f"Two circuit tensors:")
    print(f"  Circuit 0: inds {layer.circuit_tensors[0].inds}, params {layer.circuit_tensors[0].learnable_params}")
    print(f"  Circuit 1: inds {layer.circuit_tensors[1].inds}, params {layer.circuit_tensors[1].learnable_params}")
    print(f"  Total circuit params: {layer.num_circuit_params}")
    
    layer.initialize_circuit_params(method="random", seed=123)
    
    # Forward pass
    result = layer()
    print(f"\nForward result: {result.item():.4f}")


if __name__ == "__main__":
    example_basic()
    example_training()
    example_target_state()
    example_two_circuits()
