"""Demonstrating the unique advantage of HEinsum contractions.

Comparisons:
1. HEinsum with QUANTUM tensors - quantum feature encoders in tensor network
2. HEinsum with CLASSICAL tensors - learned classical encoders in same tensor network
3. Sequential quantum approach - quantum features fed to MLP (no TN interactions)
4. Classical MLP - baseline

Datasets:
- Two Moons (nonlinear boundary)
- Circles (concentric rings)  
- XOR (parity problem)
"""

import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit.circuit import QuantumCircuit, Parameter

from qtpu.heinsum import HEinsum
from qtpu.tensor import QuantumTensor, ISwitch, CTensor
from qtpu.runtime import HEinsumContractor


# =============================================================================
# Dataset Generation
# =============================================================================

def make_xor(n_samples=200, noise=0.1, random_state=42):
    """Create XOR dataset."""
    np.random.seed(random_state)
    n_per_quadrant = n_samples // 4
    
    X = []
    y = []
    
    # Quadrant (0,0) -> label 0
    X.append(np.random.randn(n_per_quadrant, 2) * 0.3 + np.array([-1, -1]))
    y.append(np.zeros(n_per_quadrant))
    
    # Quadrant (1,1) -> label 0  
    X.append(np.random.randn(n_per_quadrant, 2) * 0.3 + np.array([1, 1]))
    y.append(np.zeros(n_per_quadrant))
    
    # Quadrant (0,1) -> label 1
    X.append(np.random.randn(n_per_quadrant, 2) * 0.3 + np.array([-1, 1]))
    y.append(np.ones(n_per_quadrant))
    
    # Quadrant (1,0) -> label 1
    X.append(np.random.randn(n_per_quadrant, 2) * 0.3 + np.array([1, -1]))
    y.append(np.ones(n_per_quadrant))
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def prepare_dataset(name: str, n_samples=200):
    """Generate and prepare a dataset."""
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    elif name == "xor":
        X, y = make_xor(n_samples=n_samples, noise=0.1, random_state=42)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Scale to [0, pi]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = (X_scaled - X_scaled.min()) / (X_scaled.max() - X_scaled.min()) * np.pi
    
    return train_test_split(X_normalized, y.astype(float), test_size=0.3, random_state=42)


# =============================================================================
# Quantum Tensor Creation
# =============================================================================

def create_quantum_encoder(X: np.ndarray, feature_idx: int, basis_name: str, n_layers: int = 2) -> QuantumTensor:
    """Create a quantum encoder for specific features using ISwitch for batch.
    
    All encoders share the same 'batch' index but have different basis indices.
    """
    n_samples = X.shape[0]
    
    # Parameters - use shared batch name, unique basis name
    batch_param = Parameter("batch")  # Same for all - so batch dims align
    basis_param = Parameter(basis_name)  # Unique - for connecting via CTensors
    
    def make_feature_circuit(b: int) -> QuantumCircuit:
        """Feature map for sample b."""
        qc = QuantumCircuit(2)
        for layer in range(n_layers):
            qc.ry(X[b, feature_idx] + layer * 0.5, 0)
            qc.ry(X[b, feature_idx] + layer * 0.3, 1)
            qc.cx(0, 1)
        return qc
    
    def make_basis_circuit(k: int) -> QuantumCircuit:
        """Measurement rotation for basis k."""
        qc = QuantumCircuit(2)
        angle = k * np.pi / 4  # 4 basis states
        qc.ry(angle, 0)
        qc.ry(angle, 1)
        return qc
    
    # Main circuit
    qc = QuantumCircuit(2, 2)
    qc.append(ISwitch(batch_param, 2, n_samples, make_feature_circuit), [0, 1])
    qc.append(ISwitch(basis_param, 2, 4, make_basis_circuit), [0, 1])  # 4 basis states
    qc.measure([0, 1], [0, 1])
    
    return QuantumTensor(qc)


def create_joint_quantum_encoder(X: np.ndarray, basis_name: str = "basis_joint") -> QuantumTensor:
    """Create a quantum encoder that combines both features."""
    n_samples = X.shape[0]
    
    batch_param = Parameter("batch")  # Same as other encoders
    basis_param = Parameter(basis_name)
    
    def make_feature_circuit(b: int) -> QuantumCircuit:
        """Joint feature map for sample b."""
        qc = QuantumCircuit(2)
        qc.ry(X[b, 0], 0)
        qc.ry(X[b, 1], 1)
        qc.cx(0, 1)
        qc.rz(X[b, 0] * X[b, 1], 1)  # Product feature
        qc.cx(0, 1)
        return qc
    
    def make_basis_circuit(k: int) -> QuantumCircuit:
        """Measurement rotation for basis k."""
        qc = QuantumCircuit(2)
        angle = k * np.pi / 4
        qc.ry(angle, 0)
        qc.ry(angle, 1)
        return qc
    
    qc = QuantumCircuit(2, 2)
    qc.append(ISwitch(batch_param, 2, n_samples, make_feature_circuit), [0, 1])
    qc.append(ISwitch(basis_param, 2, 4, make_basis_circuit), [0, 1])
    qc.measure([0, 1], [0, 1])
    
    return QuantumTensor(qc)


# =============================================================================
# HEinsum with QUANTUM Tensors
# =============================================================================

def train_quantum_heinsum(X_train, y_train, X_test, y_test, epochs=200, lr=0.1):
    """HEinsum with quantum feature encoders.
    
    Architecture: q1[batch,i] * W1[i,j] * q2[batch,j] * W2[j,k] * q3[batch,k] -> [batch]
    
    All q tensors share batch dimension, but have distinct basis dimensions.
    CTensors connect the basis dimensions.
    """
    # Create quantum encoders - same batch, different basis indices
    q1 = create_quantum_encoder(X_train, feature_idx=0, basis_name="basis_0", n_layers=2)
    q2 = create_quantum_encoder(X_train, feature_idx=1, basis_name="basis_1", n_layers=2)
    q3 = create_joint_quantum_encoder(X_train, basis_name="basis_2")
    
    q1_test = create_quantum_encoder(X_test, feature_idx=0, basis_name="basis_0", n_layers=2)
    q2_test = create_quantum_encoder(X_test, feature_idx=1, basis_name="basis_1", n_layers=2)
    q3_test = create_joint_quantum_encoder(X_test, basis_name="basis_2")
    
    basis_dim = 4
    
    # Classical weight tensors connect basis dimensions
    # q1[batch, basis_0] * W1[basis_0, basis_1] * q2[batch, basis_1] * W2[basis_1, basis_2] * q3[batch, basis_2]
    W1_data = torch.randn(basis_dim, basis_dim, dtype=torch.float64) * 0.1
    W1_data.requires_grad_(True)
    W2_data = torch.randn(basis_dim, basis_dim, dtype=torch.float64) * 0.1
    W2_data.requires_grad_(True)
    
    W1 = CTensor(W1_data, inds=("basis_0", "basis_1"))
    W2 = CTensor(W2_data, inds=("basis_1", "basis_2"))
    
    n_params = W1_data.numel() + W2_data.numel()
    
    # Build HEinsum - contract over all basis dims, keep batch
    heinsum_train = HEinsum(
        qtensors=[q1, q2, q3],
        ctensors=[W1, W2],
        input_tensors=[],
        output_inds=("batch",),
    )
    
    heinsum_test = HEinsum(
        qtensors=[q1_test, q2_test, q3_test],
        ctensors=[W1, W2],
        input_tensors=[],
        output_inds=("batch",),
    )
    
    contractor_train = HEinsumContractor(heinsum_train)
    contractor_train.prepare(opt_kwargs={"max_repeats": 16, "progbar": False})
    contractor_train.precompute_quantum()
    
    contractor_test = HEinsumContractor(heinsum_test)
    contractor_test.prepare(opt_kwargs={"max_repeats": 16, "progbar": False})
    contractor_test.precompute_quantum()
    
    y_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    optimizer = torch.optim.Adam([W1_data, W2_data], lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        logits = contractor_train.contract(input_tensors=[], circuit_params={})
        probs = torch.sigmoid(logits)
        
        loss = torch.nn.functional.binary_cross_entropy(probs, y_t)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_test_logits = contractor_test.contract(input_tensors=[], circuit_params={})
        final_test_probs = torch.sigmoid(final_test_logits)
        final_test_acc = ((final_test_probs > 0.5).float() == y_test_t).float().mean()
    
    return final_test_acc.item(), n_params


# =============================================================================
# HEinsum with CLASSICAL Tensors (same tensor network structure)
# =============================================================================

def train_classical_heinsum(X_train, y_train, X_test, y_test, epochs=200, lr=0.1):
    """HEinsum with classical feature encoders (no quantum).
    
    Same tensor network structure:
    f1[batch,i] * W1[i,j] * f2[batch,j] * W2[j,k] * f3[batch,k] -> [batch]
    
    But f1, f2, f3 are LEARNED classical feature encoders instead of quantum.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    basis_dim = 4  # Same dimension as quantum
    
    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    
    # Classical feature encoders (learned)
    # f1 encodes feature 0, f2 encodes feature 1, f3 encodes both
    E1 = torch.randn(1, basis_dim, dtype=torch.float64) * 0.1
    E1.requires_grad_(True)
    E2 = torch.randn(1, basis_dim, dtype=torch.float64) * 0.1
    E2.requires_grad_(True)
    E3 = torch.randn(2, basis_dim, dtype=torch.float64) * 0.1
    E3.requires_grad_(True)
    
    # Connection weights (same as quantum version)
    W1 = torch.randn(basis_dim, basis_dim, dtype=torch.float64) * 0.1
    W1.requires_grad_(True)
    W2 = torch.randn(basis_dim, basis_dim, dtype=torch.float64) * 0.1
    W2.requires_grad_(True)
    
    n_params = E1.numel() + E2.numel() + E3.numel() + W1.numel() + W2.numel()
    
    y_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    optimizer = torch.optim.Adam([E1, E2, E3, W1, W2], lr=lr)
    
    def forward(X):
        # Classical feature encoders with nonlinearity
        f1 = torch.tanh(X[:, 0:1] @ E1)  # [batch, basis_dim]
        f2 = torch.tanh(X[:, 1:2] @ E2)  # [batch, basis_dim]
        f3 = torch.tanh(X @ E3)           # [batch, basis_dim]
        
        # Tensor network contraction (same structure as quantum version)
        # f1[b,i] * W1[i,j] * f2[b,j] * W2[j,k] * f3[b,k] -> [b]
        out = torch.einsum('bi,ij,bj,jk,bk->b', f1, W1, f2, W2, f3)
        return torch.sigmoid(out)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        probs = forward(X_train_t)
        loss = torch.nn.functional.binary_cross_entropy(probs, y_t)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_test_probs = forward(X_test_t)
        final_test_acc = ((final_test_probs > 0.5).float() == y_test_t).float().mean()
    
    return final_test_acc.item(), n_params


# =============================================================================
# Sequential Quantum (no tensor network interactions)
# =============================================================================

def train_sequential_quantum(X_train, y_train, X_test, y_test, hidden_dim=4, epochs=200, lr=0.1):
    """Extract quantum features THEN apply MLP (no tensor network interactions)."""
    
    # Same quantum encoders
    q1 = create_quantum_encoder(X_train, feature_idx=0, basis_name="basis_0", n_layers=2)
    q2 = create_quantum_encoder(X_train, feature_idx=1, basis_name="basis_1", n_layers=2)
    q3 = create_joint_quantum_encoder(X_train, basis_name="basis_2")
    
    q1_test = create_quantum_encoder(X_test, feature_idx=0, basis_name="basis_0", n_layers=2)
    q2_test = create_quantum_encoder(X_test, feature_idx=1, basis_name="basis_1", n_layers=2)
    q3_test = create_joint_quantum_encoder(X_test, basis_name="basis_2")
    
    # Extract features separately
    def extract_features(qtensor):
        heinsum = HEinsum(
            qtensors=[qtensor],
            ctensors=[],
            input_tensors=[],
            output_inds=tuple(qtensor.inds),
        )
        contractor = HEinsumContractor(heinsum)
        contractor.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
        return contractor.contract(input_tensors=[], circuit_params={}).detach()
    
    f1_train = extract_features(q1)
    f2_train = extract_features(q2)
    f3_train = extract_features(q3)
    
    f1_test = extract_features(q1_test)
    f2_test = extract_features(q2_test)
    f3_test = extract_features(q3_test)
    
    # Concatenate features
    features_train = torch.cat([f1_train, f2_train, f3_train], dim=1)
    features_test = torch.cat([f1_test, f2_test, f3_test], dim=1)
    
    total_dim = features_train.shape[1]
    
    # MLP classifier
    W1 = torch.randn(total_dim, hidden_dim, dtype=torch.float64) * 0.1
    W1.requires_grad_(True)
    b1 = torch.zeros(hidden_dim, dtype=torch.float64, requires_grad=True)
    W2 = torch.randn(hidden_dim, dtype=torch.float64) * 0.1
    W2.requires_grad_(True)
    b2 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    
    n_params = W1.numel() + b1.numel() + W2.numel() + b2.numel()
    
    y_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    optimizer = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        hidden = torch.tanh(features_train @ W1 + b1)
        logits = hidden @ W2 + b2
        probs = torch.sigmoid(logits)
        
        loss = torch.nn.functional.binary_cross_entropy(probs, y_t)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        hidden_test = torch.tanh(features_test @ W1 + b1)
        final_test_logits = hidden_test @ W2 + b2
        final_test_probs = torch.sigmoid(final_test_logits)
        final_test_acc = ((final_test_probs > 0.5).float() == y_test_t).float().mean()
    
    return final_test_acc.item(), n_params


# =============================================================================
# Classical MLP (baseline)
# =============================================================================

def train_classical_mlp(X_train, y_train, X_test, y_test, hidden_dim=16, epochs=200, lr=0.1):
    """Pure classical MLP baseline."""
    
    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    n_features = X_train.shape[1]
    
    W1 = torch.randn(n_features, hidden_dim, dtype=torch.float64) * 0.1
    W1.requires_grad_(True)
    b1 = torch.zeros(hidden_dim, dtype=torch.float64, requires_grad=True)
    
    W2 = torch.randn(hidden_dim, dtype=torch.float64) * 0.1
    W2.requires_grad_(True)
    b2 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    
    n_params = W1.numel() + b1.numel() + W2.numel() + 1
    
    optimizer = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
    
    def forward(X):
        h = torch.tanh(X @ W1 + b1)
        return torch.sigmoid(h @ W2 + b2)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        probs = forward(X_t)
        loss = torch.nn.functional.binary_cross_entropy(probs, y_t)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_test_probs = forward(X_test_t)
        final_test_acc = ((final_test_probs > 0.5).float() == y_test_t).float().mean()
    
    return final_test_acc.item(), n_params


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison(dataset_name: str, n_samples=200, epochs=200, lr=0.1):
    """Run all methods on a dataset."""
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    X_train, X_test, y_train, y_test = prepare_dataset(dataset_name, n_samples)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # 1. Quantum HEinsum
    print("\n[1] Quantum HEinsum (quantum features in tensor network)...")
    acc, params = train_quantum_heinsum(X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)
    results["Quantum HEinsum"] = (acc, params)
    print(f"    Accuracy: {acc:.3f}, Params: {params}")
    
    # 2. Classical HEinsum (same TN structure, classical features)
    print("\n[2] Classical HEinsum (classical features in same tensor network)...")
    acc, params = train_classical_heinsum(X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)
    results["Classical HEinsum"] = (acc, params)
    print(f"    Accuracy: {acc:.3f}, Params: {params}")
    
    # 3. Sequential Quantum
    print("\n[3] Sequential Quantum (quantum features -> MLP, no TN)...")
    acc, params = train_sequential_quantum(X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)
    results["Sequential Q->MLP"] = (acc, params)
    print(f"    Accuracy: {acc:.3f}, Params: {params}")
    
    # 4. Classical MLP
    print("\n[4] Classical MLP (baseline)...")
    acc, params = train_classical_mlp(X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)
    results["Classical MLP"] = (acc, params)
    print(f"    Accuracy: {acc:.3f}, Params: {params}")
    
    return results


def main():
    print("="*70)
    print("HEINSUM ADVANTAGE: Quantum vs Classical Tensor Networks")
    print("="*70)
    print("""
We compare 4 approaches:
1. Quantum HEinsum  - quantum feature encoders in tensor network
2. Classical HEinsum - same tensor network, but classical feature encoders
3. Sequential Q->MLP - quantum features fed to MLP (no TN interactions)
4. Classical MLP     - baseline

Key question: Does the tensor network structure help? Do quantum features help?
""")
    
    all_results = {}
    
    for dataset in ["moons", "circles", "xor"]:
        results = run_comparison(dataset, n_samples=200, epochs=200, lr=0.1)
        all_results[dataset] = results
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Test Accuracy by Dataset and Method")
    print("="*70)
    
    methods = ["Quantum HEinsum", "Classical HEinsum", "Sequential Q->MLP", "Classical MLP"]
    datasets = ["moons", "circles", "xor"]
    
    # Header
    print(f"\n{'Method':<20}", end="")
    for ds in datasets:
        print(f"{ds:>12}", end="")
    print(f"{'Avg':>12}")
    print("-"*70)
    
    # Rows
    for method in methods:
        print(f"{method:<20}", end="")
        accs = []
        for ds in datasets:
            acc = all_results[ds][method][0]
            accs.append(acc)
            print(f"{acc:>12.3f}", end="")
        print(f"{np.mean(accs):>12.3f}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    print("""
KEY INSIGHTS:

1. TENSOR NETWORK STRUCTURE:
   Compare Classical HEinsum vs Classical MLP - both use classical features,
   but HEinsum uses tensor network contractions. If Classical HEinsum wins,
   the TN structure itself provides benefit.

2. QUANTUM ADVANTAGE:
   Compare Quantum HEinsum vs Classical HEinsum - same TN structure,
   different feature encoders. If Quantum wins, quantum features add value.

3. TN vs SEQUENTIAL:
   Compare Quantum HEinsum vs Sequential Q->MLP - both use quantum features,
   but HEinsum contracts them together while Sequential concatenates.
   If HEinsum wins, the TN interactions matter.

The einsum contraction q1*W1*q2*W2*q3 creates multiplicative (degree-3)
interactions between features, while sequential is additive (degree-1).
""")


if __name__ == "__main__":
    main()
