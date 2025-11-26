"""Iris dataset classification: Quantum+Classical hybrid vs Pure Classical.

Compares:
1. Pure Classical: Linear classifier on raw features
2. Quantum Hybrid: Quantum feature map + classical weights via HEinsum

The quantum feature map:
- ISwitch for batch dimension: encodes X[batch, :] via Ry rotations
- ISwitch for measurement basis: rotates before measurement to extract different features
- Each measured qubit returns <Z> expectation value

This shows quantum feature encoding can provide expressive features.
"""

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit.circuit import QuantumCircuit, Parameter

from qtpu.heinsum import HEinsum
from qtpu.tensor import QuantumTensor, TensorSpec, ISwitch, CTensor
from qtpu.runtime import HEinsumContractor


def load_iris_multiclass(test_size=0.3, seed=42):
    """Load Iris dataset (3-class classification)."""
    iris = load_iris()
    X = iris.data  # (150, 4)
    y = iris.target  # 0, 1, 2
    
    # Normalize features to [0, pi] for quantum encoding
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = (X_scaled - X_scaled.min()) / (X_scaled.max() - X_scaled.min()) * np.pi
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def load_iris_binary(test_size=0.3, seed=42):
    """Load Iris dataset (binary: setosa vs non-setosa)."""
    iris = load_iris()
    X = iris.data  # (150, 4)
    y = (iris.target != 0).astype(float)  # Binary: setosa (0) vs others (1)
    
    # Normalize features to [0, pi] for quantum encoding
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = (X_scaled - X_scaled.min()) / (X_scaled.max() - X_scaled.min()) * np.pi
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def create_quantum_feature_map(X: np.ndarray, n_qubits: int = 4, n_basis: int = 4) -> QuantumTensor:
    """Create quantum feature encoder with ISwitch for batch and measurement basis.
    
    Architecture:
        - ISwitch(batch): Encodes X[batch, :] into quantum state via Ry rotations + entanglement
        - ISwitch(basis): Applies different basis rotations before measurement
        - Measures all qubits -> returns <Z> for each qubit
    
    Args:
        X: Input data of shape (batch_size, n_features)
        n_qubits: Number of qubits (should match n_features)
        n_basis: Number of different measurement bases
    
    Returns:
        QuantumTensor with shape (batch_size, n_basis) 
        Output is sum of <Z> expectations across qubits for each basis
    """
    batch_size, n_features = X.shape
    assert n_features == n_qubits, f"Features ({n_features}) must match qubits ({n_qubits})"
    
    batch_idx = Parameter("batch")
    basis_idx = Parameter("basis")
    
    # ISwitch for data encoding
    def make_encoding_circuit(b: int) -> QuantumCircuit:
        """Encode X[b, :] into entangled quantum state."""
        qc = QuantumCircuit(n_qubits)
        
        # Layer 1: Feature encoding
        for i in range(n_qubits):
            qc.ry(X[b, i], i)
        
        # Entangling layer
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)  # Circular entanglement
        
        # Layer 2: More feature encoding (data re-uploading)
        for i in range(n_qubits):
            qc.ry(X[b, i] * 0.5, i)
        
        return qc
    
    # ISwitch for measurement basis rotations
    def make_basis_circuit(basis: int) -> QuantumCircuit:
        """Apply basis rotation before measurement."""
        qc = QuantumCircuit(n_qubits)
        
        if basis == 0:
            # Z basis (no rotation)
            pass
        elif basis == 1:
            # X basis
            for i in range(n_qubits):
                qc.h(i)
        elif basis == 2:
            # Y basis
            for i in range(n_qubits):
                qc.sdg(i)
                qc.h(i)
        else:
            # Mixed rotations
            for i in range(n_qubits):
                angle = (basis * (i + 1)) * np.pi / (n_basis * n_qubits)
                qc.ry(angle, i)
        
        return qc
    
    # Build the full circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Data encoding
    qc.append(ISwitch(batch_idx, n_qubits, batch_size, make_encoding_circuit), range(n_qubits))
    
    # Basis rotation before measurement
    qc.append(ISwitch(basis_idx, n_qubits, n_basis, make_basis_circuit), range(n_qubits))
    
    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))
    
    return QuantumTensor(qc)


def create_simple_quantum_features(X: np.ndarray, n_qubits: int = 2, n_basis: int = 2) -> QuantumTensor:
    """Simpler quantum features using fewer qubits with basis ISwitch.
    
    Args:
        X: Input data of shape (batch_size, 4)
        n_qubits: Number of qubits
        n_basis: Number of measurement bases
    
    Returns:
        QuantumTensor with shape (batch_size, n_basis)
    """
    batch_size, n_features = X.shape
    batch_idx = Parameter("batch")
    basis_idx = Parameter("basis")
    
    # Data encoding
    def make_encoding(b: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        
        # Encode all 4 features into 2 qubits
        qc.ry(X[b, 0], 0)
        qc.ry(X[b, 1], 1)
        qc.cx(0, 1)
        qc.ry(X[b, 2], 0)
        qc.ry(X[b, 3], 1)
        
        return qc
    
    # Measurement basis
    def make_basis(basis: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        if basis == 0:
            pass  # Z basis
        else:
            # Rotate by different angles
            for i in range(n_qubits):
                qc.ry(basis * np.pi / n_basis, i)
        return qc
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.append(ISwitch(batch_idx, n_qubits, batch_size, make_encoding), range(n_qubits))
    qc.append(ISwitch(basis_idx, n_qubits, n_basis, make_basis), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    
    return QuantumTensor(qc)


def train_pure_classical(X_train, y_train, X_test, y_test, n_features_out=4, epochs=200, lr=0.1):
    """Train pure classical linear classifier.
    
    Architecture: X[batch, 4] @ W[4] + bias -> logits[batch]
    
    Same number of parameters as quantum hybrid for fair comparison.
    """
    print("\n" + "="*60)
    print("PURE CLASSICAL BASELINE")
    print("="*60)
    
    batch_size, n_features = X_train.shape
    
    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    # Trainable weights - same number as quantum version
    W = torch.randn(n_features, dtype=torch.float64) * 0.1
    W.requires_grad_(True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    
    n_params = W.numel() + 1
    print(f"Architecture: X[batch,{n_features}] @ W[{n_features}] + bias -> logits[batch]")
    print(f"Trainable parameters: {n_params}")
    
    optimizer = torch.optim.Adam([W, bias], lr=lr)
    
    def forward(X):
        logits = X @ W + bias  # [batch]
        return torch.sigmoid(logits)
    
    print("\nTraining...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        probs = forward(X_t)
        loss = torch.nn.functional.binary_cross_entropy(probs, y_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 40 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                train_acc = ((probs > 0.5).float() == y_t).float().mean()
                test_probs = forward(X_test_t)
                test_acc = ((test_probs > 0.5).float() == y_test_t).float().mean()
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
    
    with torch.no_grad():
        final_test_probs = forward(X_test_t)
        final_test_acc = ((final_test_probs > 0.5).float() == y_test_t).float().mean()
    
    return final_test_acc.item(), n_params


def train_quantum_hybrid(X_train, y_train, X_test, y_test, n_basis=4, epochs=200, lr=0.1):
    """Train quantum+classical hybrid using HEinsum.
    
    Architecture: 
        ISwitch(batch) x ISwitch(basis) -> q[batch, basis]
        q[batch, basis] @ W[basis] + bias -> logits[batch]
    
    The quantum tensor gives us (batch_size, n_basis) features from Z expectations.
    """
    print("\n" + "="*60)
    print("QUANTUM + CLASSICAL HYBRID")
    print("="*60)
    
    n_qubits = 4  # Use all 4 features
    
    # Create quantum feature maps for train and test
    print("Creating quantum feature encoders...")
    qtensor_train = create_quantum_feature_map(X_train, n_qubits=n_qubits, n_basis=n_basis)
    qtensor_test = create_quantum_feature_map(X_test, n_qubits=n_qubits, n_basis=n_basis)
    
    print(f"Quantum tensor train shape: {qtensor_train.shape}")
    print(f"Quantum tensor train inds: {qtensor_train.inds}")
    
    # Trainable classical weights - same number as classical baseline
    W = torch.randn(n_basis, dtype=torch.float64) * 0.1
    W.requires_grad_(True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    
    n_params = W.numel() + 1
    print(f"Architecture: q[batch,{n_basis}] @ W[{n_basis}] + bias -> logits[batch]")
    print(f"Trainable parameters: {n_params}")
    print(f"Quantum feature dim: {n_basis} (from {n_basis} measurement bases)")
    
    # HEinsum for quantum features: q[batch, basis] -> output[batch, basis]
    heinsum_train = HEinsum(
        qtensors=[qtensor_train],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch", "basis"),
    )
    
    heinsum_test = HEinsum(
        qtensors=[qtensor_test],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch", "basis"),
    )
    
    print("Preparing contractors...")
    contractor_train = HEinsumContractor(heinsum_train)
    contractor_train.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    contractor_test = HEinsumContractor(heinsum_test)
    contractor_test.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    # Get quantum features (fixed, no quantum params to train)
    print("Extracting quantum features...")
    q_train = contractor_train.contract(input_tensors=[], circuit_params={}).detach()
    q_test = contractor_test.contract(input_tensors=[], circuit_params={}).detach()
    
    print(f"Quantum features train shape: {q_train.shape}")
    print(f"Quantum features test shape: {q_test.shape}")
    
    y_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    optimizer = torch.optim.Adam([W, bias], lr=lr)
    
    def forward(q_features):
        logits = q_features @ W + bias  # [batch]
        return torch.sigmoid(logits)
    
    print("\nTraining...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        probs = forward(q_train)
        loss = torch.nn.functional.binary_cross_entropy(probs, y_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 40 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                train_acc = ((probs > 0.5).float() == y_t).float().mean()
                test_probs = forward(q_test)
                test_acc = ((test_probs > 0.5).float() == y_test_t).float().mean()
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
    
    with torch.no_grad():
        final_test_probs = forward(q_test)
        final_test_acc = ((final_test_probs > 0.5).float() == y_test_t).float().mean()
    
    return final_test_acc.item(), n_params


def train_quantum_hybrid_heinsum(X_train, y_train, X_test, y_test, n_basis=4, epochs=200, lr=0.1):
    """Train quantum+classical hybrid using HEinsum with CTensor weights.
    
    The contraction q[batch, basis] @ W[basis] happens inside HEinsum.
    Since quantum features are fixed, we cache them for efficiency.
    """
    print("\n" + "="*60)
    print("QUANTUM + CLASSICAL HYBRID (Full HEinsum)")
    print("="*60)
    
    n_qubits = 4
    
    print("Creating quantum feature encoders...")
    qtensor_train = create_quantum_feature_map(X_train, n_qubits=n_qubits, n_basis=n_basis)
    qtensor_test = create_quantum_feature_map(X_test, n_qubits=n_qubits, n_basis=n_basis)
    
    # First extract quantum features (since they're fixed)
    heinsum_q_train = HEinsum(
        qtensors=[qtensor_train],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch", "basis"),
    )
    heinsum_q_test = HEinsum(
        qtensors=[qtensor_test],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch", "basis"),
    )
    
    print("Extracting quantum features...")
    contractor_q_train = HEinsumContractor(heinsum_q_train)
    contractor_q_train.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    q_train = contractor_q_train.contract(input_tensors=[], circuit_params={}).detach()
    
    contractor_q_test = HEinsumContractor(heinsum_q_test)
    contractor_q_test.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    q_test = contractor_q_test.contract(input_tensors=[], circuit_params={}).detach()
    
    print(f"Quantum features train shape: {q_train.shape}")
    print(f"Quantum features test shape: {q_test.shape}")
    
    # Now set up HEinsum for contraction with trainable weights
    # q[batch, basis] @ W[basis] -> output[batch]
    q_spec = TensorSpec((len(X_train), n_basis), ("batch", "basis"))
    q_test_spec = TensorSpec((len(X_test), n_basis), ("batch", "basis"))
    
    W_data = torch.randn(n_basis, dtype=torch.float64) * 0.1
    W_data.requires_grad_(True)
    W = CTensor(W_data, inds=("basis",))
    
    n_params = W_data.numel()
    print(f"Architecture: q[batch,{n_basis}] @ W[{n_basis}] -> logits[batch]")
    print(f"Trainable parameters: {n_params}")
    
    # HEinsum with input_tensor for q and CTensor for W
    heinsum_train = HEinsum(
        qtensors=[],
        ctensors=[W],
        input_tensors=[q_spec],
        output_inds=("batch",),
    )
    heinsum_test = HEinsum(
        qtensors=[],
        ctensors=[W],
        input_tensors=[q_test_spec],
        output_inds=("batch",),
    )
    
    print("Preparing contractors...")
    contractor_train = HEinsumContractor(heinsum_train)
    contractor_train.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    contractor_test = HEinsumContractor(heinsum_test)
    contractor_test.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    y_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    optimizer = torch.optim.Adam([W_data], lr=lr)
    
    print("\nTraining...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward through HEinsum: q @ W
        logits = contractor_train.contract(input_tensors=[q_train], circuit_params={})
        probs = torch.sigmoid(logits)
        
        loss = torch.nn.functional.binary_cross_entropy(probs, y_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 40 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                train_acc = ((probs > 0.5).float() == y_t).float().mean()
                
                test_logits = contractor_test.contract(input_tensors=[q_test], circuit_params={})
                test_probs = torch.sigmoid(test_logits)
                test_acc = ((test_probs > 0.5).float() == y_test_t).float().mean()
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
    
    with torch.no_grad():
        test_logits = contractor_test.contract(input_tensors=[q_test], circuit_params={})
        final_test_probs = torch.sigmoid(test_logits)
        final_test_acc = ((final_test_probs > 0.5).float() == y_test_t).float().mean()
    
    return final_test_acc.item(), n_params


def train_pure_classical_multiclass(X_train, y_train, X_test, y_test, n_classes=3, epochs=300, lr=0.1):
    """Train pure classical multiclass classifier.
    
    Architecture: X[batch, 4] @ W[4, n_classes] + bias[n_classes] -> logits[batch, n_classes]
    """
    print("\n" + "="*60)
    print("PURE CLASSICAL BASELINE (Multiclass)")
    print("="*60)
    
    batch_size, n_features = X_train.shape
    
    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # Trainable weights
    W = torch.randn(n_features, n_classes, dtype=torch.float64) * 0.1
    W.requires_grad_(True)
    bias = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    
    n_params = W.numel() + bias.numel()
    print(f"Architecture: X[batch,{n_features}] @ W[{n_features},{n_classes}] + bias[{n_classes}] -> logits[batch,{n_classes}]")
    print(f"Trainable parameters: {n_params}")
    
    optimizer = torch.optim.Adam([W, bias], lr=lr)
    
    def forward(X):
        logits = X @ W + bias  # [batch, n_classes]
        return logits
    
    print("\nTraining...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        logits = forward(X_t)
        loss = torch.nn.functional.cross_entropy(logits, y_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 60 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                train_preds = logits.argmax(dim=1)
                train_acc = (train_preds == y_t).float().mean()
                test_logits = forward(X_test_t)
                test_preds = test_logits.argmax(dim=1)
                test_acc = (test_preds == y_test_t).float().mean()
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
    
    with torch.no_grad():
        final_test_logits = forward(X_test_t)
        final_test_preds = final_test_logits.argmax(dim=1)
        final_test_acc = (final_test_preds == y_test_t).float().mean()
    
    return final_test_acc.item(), n_params


def train_quantum_hybrid_multiclass(X_train, y_train, X_test, y_test, n_classes=3, n_basis=4, epochs=300, lr=0.1):
    """Train quantum+classical hybrid multiclass classifier.
    
    Architecture: 
        ISwitch(batch) x ISwitch(basis) -> q[batch, basis]
        q[batch, basis] @ W[basis, n_classes] + bias[n_classes] -> logits[batch, n_classes]
    """
    print("\n" + "="*60)
    print("QUANTUM + CLASSICAL HYBRID (Multiclass)")
    print("="*60)
    
    n_qubits = 4  # Use all 4 features
    
    # Create quantum feature maps
    print("Creating quantum feature encoders...")
    qtensor_train = create_quantum_feature_map(X_train, n_qubits=n_qubits, n_basis=n_basis)
    qtensor_test = create_quantum_feature_map(X_test, n_qubits=n_qubits, n_basis=n_basis)
    
    print(f"Quantum tensor train shape: {qtensor_train.shape}")
    print(f"Quantum tensor train inds: {qtensor_train.inds}")
    
    # Trainable classical weights
    W = torch.randn(n_basis, n_classes, dtype=torch.float64) * 0.1
    W.requires_grad_(True)
    bias = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    
    n_params = W.numel() + bias.numel()
    print(f"Architecture: q[batch,{n_basis}] @ W[{n_basis},{n_classes}] + bias[{n_classes}] -> logits[batch,{n_classes}]")
    print(f"Trainable parameters: {n_params}")
    print(f"Quantum feature dim: {n_basis} (from {n_basis} measurement bases)")
    
    # HEinsum for quantum features: q[batch, basis]
    heinsum_train = HEinsum(
        qtensors=[qtensor_train],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch", "basis"),
    )
    heinsum_test = HEinsum(
        qtensors=[qtensor_test],
        ctensors=[],
        input_tensors=[],
        output_inds=("batch", "basis"),
    )
    
    print("Preparing contractors...")
    contractor_train = HEinsumContractor(heinsum_train)
    contractor_train.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    contractor_test = HEinsumContractor(heinsum_test)
    contractor_test.prepare(opt_kwargs={"max_repeats": 8, "progbar": False})
    
    # Extract quantum features (fixed)
    print("Extracting quantum features...")
    q_train = contractor_train.contract(input_tensors=[], circuit_params={}).detach()
    q_test = contractor_test.contract(input_tensors=[], circuit_params={}).detach()
    
    print(f"Quantum features train shape: {q_train.shape}")
    print(f"Quantum features test shape: {q_test.shape}")
    
    y_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    optimizer = torch.optim.Adam([W, bias], lr=lr)
    
    def forward(q_features):
        logits = q_features @ W + bias  # [batch, n_classes]
        return logits
    
    print("\nTraining...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        logits = forward(q_train)
        loss = torch.nn.functional.cross_entropy(logits, y_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 60 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                train_preds = logits.argmax(dim=1)
                train_acc = (train_preds == y_t).float().mean()
                test_logits = forward(q_test)
                test_preds = test_logits.argmax(dim=1)
                test_acc = (test_preds == y_test_t).float().mean()
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
    
    with torch.no_grad():
        final_test_logits = forward(q_test)
        final_test_preds = final_test_logits.argmax(dim=1)
        final_test_acc = (final_test_preds == y_test_t).float().mean()
    
    return final_test_acc.item(), n_params


def main():
    print("="*60)
    print("IRIS CLASSIFICATION: Quantum vs Classical Comparison")
    print("="*60)
    print("\nThis comparison shows how quantum feature maps can extract")
    print("useful features from data that improve classification.")
    
    # Load data - use 3-class for harder task
    X_train, X_test, y_train, y_test = load_iris_multiclass()
    n_classes = 3
    
    print(f"\nDataset: Iris (3-class)")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: 4 (sepal/petal length/width)")
    print(f"Classes: {n_classes} (setosa, versicolor, virginica)")
    
    # Compare with same parameter count
    n_basis = 4  # Same as input features
    
    results = {}
    
    # 1. Pure classical baseline (same params)
    acc_classical, params_classical = train_pure_classical_multiclass(
        X_train, y_train, X_test, y_test, 
        n_classes=n_classes, epochs=300, lr=0.1
    )
    results["Classical (4 feat)"] = (acc_classical, params_classical)
    
    # 2. Quantum hybrid (same params, different features)
    acc_hybrid, params_hybrid = train_quantum_hybrid_multiclass(
        X_train, y_train, X_test, y_test,
        n_classes=n_classes, n_basis=n_basis, epochs=300, lr=0.1
    )
    results["Quantum (4 basis)"] = (acc_hybrid, params_hybrid)
    
    # 3. Quantum with more bases (more expressivity)
    acc_hybrid_8, params_hybrid_8 = train_quantum_hybrid_multiclass(
        X_train, y_train, X_test, y_test,
        n_classes=n_classes, n_basis=8, epochs=300, lr=0.1
    )
    results["Quantum (8 basis)"] = (acc_hybrid_8, params_hybrid_8)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<25} {'Test Accuracy':<15} {'Parameters':<10}")
    print("-"*50)
    for method, (acc, params) in results.items():
        print(f"{method:<25} {acc:.3f}           {params}")
    
    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("1. Quantum feature maps transform 4 input features into quantum features")
    print("2. Each measurement basis extracts different information via <Z> expectation")
    print("3. The quantum circuit uses entanglement + data re-uploading for expressivity")
    print("4. Classical gets raw features X[batch,4], quantum gets q[batch,n_basis]")
    print("\nNote: Iris is a simple dataset where linear classifiers already perform")
    print("      very well. Quantum advantage is more visible on harder problems")
    print("      with non-linear decision boundaries or limited labeled data.")


if __name__ == "__main__":
    main()
