"""Hybrid Quantum-Classical Tensor Network Training.

This demonstrates combining quantum circuit tensors with classical tensors
in more complex einsum expressions:

1. Quantum Feature Extractor + Classical Linear Layer
2. Multi-head Quantum Attention (multiple circuits)
3. Quantum Kernel with Classical Post-processing

The key insight: einsum naturally expresses how quantum and classical 
tensors interact, and all classical parts get gradients via autograd.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit, Parameter

from qtpu.tensor import ISwitch
from qtpu.torch import HTNLayer, ExpvalTorchEvaluator


# =============================================================================
# Helper: Create data-encoding quantum circuits
# =============================================================================

def make_encoding_circuit(
    x_values: np.ndarray,
    num_qubits: int = 1,
    num_layers: int = 2,
    batch_idx_name: str = "b",
) -> QuantumCircuit:
    """Create a variational circuit encoding data via ISwitch."""
    batch_size = len(x_values)
    qc = QuantumCircuit(num_qubits)
    
    batch_param = Parameter(batch_idx_name)
    
    def encoder(idx: int) -> QuantumCircuit:
        enc = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            enc.ry(x_values[idx] * (q + 1), q)  # Different encoding per qubit
        return enc
    
    iswitch = ISwitch(batch_param, num_qubits, batch_size, encoder)
    qc.append(iswitch, range(num_qubits))
    
    # Variational layers
    for layer in range(num_layers):
        for q in range(num_qubits):
            qc.ry(Parameter(f"theta_{layer}_{q}"), q)
            qc.rz(Parameter(f"phi_{layer}_{q}"), q)
        # Entangling
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    
    qc.measure_all()
    return qc


# =============================================================================
# Example 1: Quantum Features + Classical Linear Layer
# =============================================================================

def example_quantum_linear():
    """
    Architecture:
        Input x → [Quantum Circuit] → features (b, f) → [Linear W] → output (b,)
    
    Einsum: "bf,f->b"
        - b: batch dimension (from ISwitch)
        - f: feature dimension (multiple qubits)
        - W: learnable weight vector (f,)
    """
    print("=" * 70)
    print("Example 1: Quantum Feature Extractor + Classical Linear Layer")
    print("=" * 70)
    
    # Data
    np.random.seed(42)
    n_train = 20
    x_train = np.linspace(0, 2 * np.pi, n_train)
    y_train = np.sin(x_train) + 0.5 * np.cos(2 * x_train)  # More complex target
    
    print(f"Target: sin(x) + 0.5*cos(2x)")
    print(f"Training samples: {n_train}")
    
    # Quantum circuit with multiple qubits → multiple features
    num_qubits = 3  # 3 features from 3 qubits
    qc = make_encoding_circuit(x_train, num_qubits=num_qubits, num_layers=2)
    
    # But we need to output (batch, features) not just (batch,)
    # For this, we'll use separate circuits for each feature
    # Or measure each qubit separately...
    
    # Simpler: use 3 single-qubit circuits as 3 features
    circuits = []
    for f in range(3):
        qc_f = QuantumCircuit(1)
        batch_param = Parameter("b")
        
        def make_encoder(feature_idx):
            def encoder(idx):
                enc = QuantumCircuit(1)
                enc.ry(x_train[idx] * (feature_idx + 1), 0)
                return enc
            return encoder
        
        iswitch = ISwitch(batch_param, 1, n_train, make_encoder(f))
        qc_f.append(iswitch, [0])
        qc_f.ry(Parameter(f"theta_{f}"), 0)
        qc_f.rz(Parameter(f"phi_{f}"), 0)
        qc_f.measure_all()
        circuits.append(qc_f)
    
    evaluator = ExpvalTorchEvaluator()
    
    # Classical weight vector (learnable)
    W = torch.randn(3, dtype=torch.float64) * 0.1
    bias = torch.zeros(1, dtype=torch.float64)
    
    # Einsum: 3 quantum tensors (b each) contracted with weights
    # "b,b,b,f->b" where we stack the 3 circuits and contract with W
    # Actually simpler: evaluate separately and combine
    
    # Let's use a single circuit with output dimension for features
    # Use 3 circuits: Q0(b), Q1(b), Q2(b) and W(f=3)
    # Contract: sum_f Q_f[b] * W[f] → output[b]
    
    # With current HTNLayer, we do: "a,b,c,abc->abc" then reduce
    # Or build manually...
    
    # Actually, let's do it cleaner with proper multi-output
    print("\nArchitecture:")
    print("  3 quantum circuits (1 qubit each) → 3 features per sample")
    print("  Classical linear: features @ W + bias → scalar output")
    print("  Einsum: 'b,b,b,f->b' (contract features with weights)")
    
    # Create combined layer
    # Q0: shape (b,), Q1: shape (b,), Q2: shape (b,), W: shape (3,)
    # We want: output[b] = Q0[b]*W[0] + Q1[b]*W[1] + Q2[b]*W[2]
    # But einsum doesn't quite work this way...
    
    # Let's be more explicit: stack quantum outputs, then matmul
    layers = []
    for qc_f in circuits:
        layer = HTNLayer(
            einsum_expr="b->b",
            circuit_tensors=[qc_f],
            evaluator=evaluator,
            learnable_circuit_params=True,
        )
        layer.initialize_circuit_params(method="random", seed=42)
        layers.append(layer)
    
    # Classical parameters
    W = torch.nn.Parameter(torch.randn(3, dtype=torch.float64) * 0.1)
    bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    
    # Combine into a module
    class HybridModel(torch.nn.Module):
        def __init__(self, quantum_layers, W, bias):
            super().__init__()
            self.quantum_layers = torch.nn.ModuleList(quantum_layers)
            self.W = W
            self.bias = bias
        
        def forward(self):
            # Get quantum features: (batch, n_features)
            features = torch.stack([layer() for layer in self.quantum_layers], dim=1)
            # Linear combination
            return features @ self.W + self.bias
    
    model = HybridModel(layers, W, bias)
    
    y_target = torch.tensor(y_train, dtype=torch.float64)
    optimizer = optim.Adam(model.parameters(), lr=0.15)
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    print("  - Quantum circuit params")
    print("  - Classical W (3,) and bias (1,)")
    
    print("\nTraining...")
    for epoch in range(80):
        optimizer.zero_grad()
        y_pred = model()
        loss = torch.mean((y_pred - y_target) ** 2)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: MSE = {loss.item():.6f}")
    
    print(f"\nFinal MSE: {loss.item():.6f}")
    print(f"Learned W: {model.W.data.numpy()}")
    print(f"Learned bias: {model.bias.data.item():.4f}")


# =============================================================================
# Example 2: Quantum Kernel + Classical Post-processing
# =============================================================================

def example_quantum_kernel():
    """
    Quantum Kernel: compute similarity between all pairs of inputs.
    
    Architecture:
        For each pair (i, j): circuit encodes both x_i and x_j
        Output: kernel matrix K[i,j] = <ψ(x_i)|ψ(x_j)>
        Then: classical layer on kernel → predictions
    
    Einsum: "ij,ij,j->i"
        - K[i,j]: quantum kernel (2D ISwitch)
        - W[i,j]: learnable attention/weights
        - y_train[j]: training labels
        - Output: predictions for each i
    """
    print("\n" + "=" * 70)
    print("Example 2: Quantum Kernel + Classical Processing")
    print("=" * 70)
    
    np.random.seed(123)
    n_samples = 12
    
    x_train = np.linspace(0, 2 * np.pi, n_samples)
    y_train = np.sin(x_train)
    
    print(f"Quantum kernel-based regression")
    print(f"  Samples: {n_samples}")
    print(f"  Kernel: K[i,j] = circuit overlap between x_i and x_j")
    
    # Create 2D ISwitch kernel circuit
    qc = QuantumCircuit(1)
    
    i_param = Parameter("i")
    j_param = Parameter("j")
    
    # Encode x_i
    def encoder_i(idx):
        enc = QuantumCircuit(1)
        enc.ry(x_train[idx], 0)
        return enc
    
    iswitch_i = ISwitch(i_param, 1, n_samples, encoder_i)
    qc.append(iswitch_i, [0])
    
    # Variational layer
    qc.ry(Parameter("theta"), 0)
    
    # "Decode" with x_j (adjoint-like)
    def encoder_j(idx):
        enc = QuantumCircuit(1)
        enc.ry(-x_train[idx], 0)  # Negative = adjoint of RY
        return enc
    
    iswitch_j = ISwitch(j_param, 1, n_samples, encoder_j)
    qc.append(iswitch_j, [0])
    
    qc.measure_all()
    
    print(f"\nCircuit creates kernel matrix K[i,j] of shape ({n_samples}, {n_samples})")
    
    evaluator = ExpvalTorchEvaluator()
    
    # HTNLayer with 2D output (kernel matrix)
    # Then contract with learnable weights and training labels
    # "ij,ij,j->i": K * W * y_train summed over j
    
    W_init = torch.ones(n_samples, n_samples, dtype=torch.float64) / n_samples
    
    layer = HTNLayer(
        einsum_expr="ij,ij,j->i",  # Kernel weighted sum
        circuit_tensors=[qc],
        evaluator=evaluator,
        classical_tensors=[W_init],  # Learnable weights
        learnable_circuit_params=True,
    )
    layer.initialize_circuit_params(method="zeros")
    
    y_target = torch.tensor(y_train, dtype=torch.float64)
    
    print("\nEinsum: 'ij,ij,j->i'")
    print("  - ij: Quantum kernel K[i,j]")
    print("  - ij: Learnable weight matrix W[i,j]") 
    print("  - j: Training labels y[j]")
    print("  - Output: predictions[i]")
    
    optimizer = optim.Adam(layer.parameters(), lr=0.1)
    
    print("\nTraining...")
    for epoch in range(60):
        optimizer.zero_grad()
        
        y_pred = layer(y_target)  # Pass y_train as input tensor
        loss = torch.mean((y_pred - y_target) ** 2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 15 == 0:
            print(f"  Epoch {epoch:3d}: MSE = {loss.item():.6f}")
    
    print(f"\nFinal MSE: {loss.item():.6f}")
    
    # Visualize kernel
    with torch.no_grad():
        # Get kernel matrix by evaluating circuit
        K = layer._evaluate_circuit_tensor_differentiable(0, layer._get_circuit_params(0))
        
    print(f"Kernel matrix shape: {K.shape}")


# =============================================================================
# Example 3: Multi-Circuit Attention Mechanism
# =============================================================================

def example_quantum_attention():
    """
    Multi-head quantum attention:
        - Query circuit Q[b,h]: batch × heads
        - Key circuit K[b,h]: batch × heads  
        - Value circuit V[b]: batch
        - Attention: softmax(Q @ K^T) @ V
    
    Simplified version using einsum.
    """
    print("\n" + "=" * 70)
    print("Example 3: Quantum Attention Mechanism")
    print("=" * 70)
    
    np.random.seed(456)
    n_samples = 10
    n_heads = 2
    
    x_train = np.linspace(0, 2 * np.pi, n_samples)
    y_train = np.sin(x_train) * np.cos(x_train / 2)
    
    print(f"Quantum attention for sequence modeling")
    print(f"  Sequence length: {n_samples}")
    print(f"  Attention heads: {n_heads}")
    
    evaluator = ExpvalTorchEvaluator()
    
    # Query circuit: (batch, heads)
    def make_qkv_circuit(name_prefix, head_idx):
        qc = QuantumCircuit(1)
        b_param = Parameter("b")
        
        def encoder(idx):
            enc = QuantumCircuit(1)
            enc.ry(x_train[idx] * (head_idx + 1), 0)
            return enc
        
        iswitch = ISwitch(b_param, 1, n_samples, encoder)
        qc.append(iswitch, [0])
        qc.ry(Parameter(f"{name_prefix}_theta_{head_idx}"), 0)
        qc.rz(Parameter(f"{name_prefix}_phi_{head_idx}"), 0)
        qc.measure_all()
        return qc
    
    # Create Q, K, V circuits for each head
    Q_circuits = [make_qkv_circuit("Q", h) for h in range(n_heads)]
    K_circuits = [make_qkv_circuit("K", h) for h in range(n_heads)]
    V_circuit = make_qkv_circuit("V", 0)
    
    # Build layers
    Q_layers = [HTNLayer("b->b", [qc], evaluator, learnable_circuit_params=True) for qc in Q_circuits]
    K_layers = [HTNLayer("b->b", [qc], evaluator, learnable_circuit_params=True) for qc in K_circuits]
    V_layer = HTNLayer("b->b", [V_circuit], evaluator, learnable_circuit_params=True)
    
    for layer in Q_layers + K_layers + [V_layer]:
        layer.initialize_circuit_params(method="random", seed=789)
    
    # Classical attention weights (learnable)
    attn_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    output_proj = torch.nn.Parameter(torch.randn(n_heads, dtype=torch.float64) * 0.1)
    
    class QuantumAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Q_layers = torch.nn.ModuleList(Q_layers)
            self.K_layers = torch.nn.ModuleList(K_layers)
            self.V_layer = V_layer
            self.attn_scale = attn_scale
            self.output_proj = output_proj
        
        def forward(self):
            # Q, K: (batch, heads), V: (batch,)
            Q = torch.stack([l() for l in self.Q_layers], dim=1)  # (b, h)
            K = torch.stack([l() for l in self.K_layers], dim=1)  # (b, h)
            V = self.V_layer()  # (b,)
            
            # Attention scores: Q @ K^T → (b, b)
            # Using einsum: "bh,ch->bc"
            scores = torch.einsum("bh,ch->bc", Q, K) * self.attn_scale
            attn_weights = torch.softmax(scores, dim=-1)  # (b, b)
            
            # Apply attention to values: (b, b) @ (b,) → (b,)
            attended = torch.einsum("bc,c->b", attn_weights, V)
            
            # Multi-head combination (simplified)
            # Actually let's just output attended directly
            return attended
    
    model = QuantumAttention()
    y_target = torch.tensor(y_train, dtype=torch.float64)
    
    print("\nArchitecture:")
    print("  Q[b,h] = Query circuits (batch × heads)")
    print("  K[b,h] = Key circuits (batch × heads)")
    print("  V[b] = Value circuit (batch)")
    print("  Attention = softmax(Q @ K^T) @ V")
    
    optimizer = optim.Adam(model.parameters(), lr=0.2)
    
    print("\nTraining...")
    for epoch in range(60):
        optimizer.zero_grad()
        y_pred = model()
        loss = torch.mean((y_pred - y_target) ** 2)
        loss.backward()
        optimizer.step()
        
        if epoch % 15 == 0:
            print(f"  Epoch {epoch:3d}: MSE = {loss.item():.6f}")
    
    print(f"\nFinal MSE: {loss.item():.6f}")


# =============================================================================
# Example 4: Full Hybrid TN with Complex Einsum
# =============================================================================

def example_complex_einsum():
    """
    Complex hybrid tensor network:
        - Quantum tensor Q[b,i]: batch × feature
        - Classical embedding E[i,d]: feature → hidden
        - Classical projection P[d]: hidden → scalar
    
    Einsum: "bi,id,d->b"
    """
    print("\n" + "=" * 70)
    print("Example 4: Complex Hybrid Einsum Network")
    print("=" * 70)
    
    np.random.seed(999)
    n_batch = 8
    n_features = 3
    n_hidden = 4
    
    x_train = np.linspace(0, 2 * np.pi, n_batch)
    y_train = np.sin(x_train) + 0.3 * np.sin(3 * x_train)
    
    print(f"Complex einsum: 'bi,id,d->b'")
    print(f"  Q[b,i]: Quantum tensor ({n_batch}, {n_features})")
    print(f"  E[i,d]: Embedding matrix ({n_features}, {n_hidden}) - learnable")
    print(f"  P[d]: Projection vector ({n_hidden},) - learnable")
    
    # Quantum circuit with 2D output: (batch, features)
    qc = QuantumCircuit(1)
    
    b_param = Parameter("b")
    i_param = Parameter("i")
    
    def encoder_b(idx):
        enc = QuantumCircuit(1)
        enc.ry(x_train[idx], 0)
        return enc
    
    def encoder_i(idx):
        enc = QuantumCircuit(1)
        enc.rz(idx * np.pi / n_features, 0)
        return enc
    
    iswitch_b = ISwitch(b_param, 1, n_batch, encoder_b)
    iswitch_i = ISwitch(i_param, 1, n_features, encoder_i)
    
    qc.append(iswitch_b, [0])
    qc.ry(Parameter("theta"), 0)
    qc.append(iswitch_i, [0])
    qc.rz(Parameter("phi"), 0)
    qc.measure_all()
    
    evaluator = ExpvalTorchEvaluator()
    
    # Classical tensors
    E_init = torch.randn(n_features, n_hidden, dtype=torch.float64) * 0.3
    P_init = torch.randn(n_hidden, dtype=torch.float64) * 0.3
    
    layer = HTNLayer(
        einsum_expr="bi,id,d->b",  # No extra input tensor needed
        circuit_tensors=[qc],
        evaluator=evaluator,
        classical_tensors=[E_init, P_init],  # Learnable E and P
        learnable_circuit_params=True,
    )
    layer.initialize_circuit_params(method="random", seed=111)
    
    print(f"\nHTNLayer structure:")
    print(f"  Circuit tensors: {layer.num_circuit_tensors}")
    print(f"  Circuit params: {layer.num_circuit_params}")
    print(f"  Learnable classical: {layer.num_learnable_classical}")
    print(f"  Input tensors: {layer.num_input_tensors}")
    
    y_target = torch.tensor(y_train, dtype=torch.float64)
    
    optimizer = optim.Adam(layer.parameters(), lr=0.2)
    
    print("\nTraining...")
    for epoch in range(60):
        optimizer.zero_grad()
        y_pred = layer()  # No input tensors needed
        loss = torch.mean((y_pred - y_target) ** 2)
        loss.backward()
        optimizer.step()
        
        if epoch % 15 == 0:
            print(f"  Epoch {epoch:3d}: MSE = {loss.item():.6f}")
    
    print(f"\nFinal MSE: {loss.item():.6f}")
    
    # Show learned classical tensors
    E = layer.get_learnable_classical(0)
    P = layer.get_learnable_classical(1)
    print(f"\nLearned E shape: {E.shape}, norm: {E.norm().item():.4f}")
    print(f"Learned P shape: {P.shape}, norm: {P.norm().item():.4f}")


if __name__ == "__main__":
    example_quantum_linear()
    example_quantum_kernel()
    example_quantum_attention()
    example_complex_einsum()
