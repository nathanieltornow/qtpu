"""
MNIST Classification with Quantum Kernel Network (HEinsum)
==========================================================

A hierarchical quantum feature network for MNIST digit classification.

Architecture:
  - 4 quantum feature extractors (one per 2x2 image region)
  - Each extracts a [batch, basis] tensor from its image patch
  - Quantum features are extracted ONCE and cached
  - Classical MLP trained on concatenated quantum features
  - Final output: binary classification (e.g., 0 vs 1)

This demonstrates HEinsum's ability to:
  1. Broadcast a single parameterized circuit over many inputs (via ISwitch)
  2. Extract rich quantum features via multiple measurement bases
  3. Efficient training by caching fixed quantum features
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit, Parameter, ClassicalRegister
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qtpu.core import HEinsum, QuantumTensor, CTensor, ISwitch
from qtpu.runtime import HEinsumRuntime


# =============================================================================
# Data Loading
# =============================================================================

def load_mnist_binary(digit_a: int = 0, digit_b: int = 1, n_samples: int = 200, seed: int = 42):
    """Load MNIST and filter to two digits for binary classification.
    
    Also downsamples to 4x4 for tractable quantum encoding.
    """
    print(f"Loading MNIST ({digit_a} vs {digit_b})...")
    
    np.random.seed(seed)
    
    # Load a small subset of MNIST
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
    
    # Reshape to 28x28 and downsample to 4x4
    X = X.reshape(-1, 28, 28)
    X_down = np.zeros((len(X), 4, 4))
    for i in range(4):
        for j in range(4):
            # Average pool 7x7 regions
            X_down[:, i, j] = X[:, i*7:(i+1)*7, j*7:(j+1)*7].mean(axis=(1, 2))
    
    # Normalize to [0, pi] for angle encoding
    X_down = X_down.reshape(-1, 16)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_down = scaler.fit_transform(X_down)
    
    return X_down, y


def extract_patches(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract 4 patches (2x2 grid) from 4x4 images.
    
    Patch layout:
      [P1] [P2]
      [P3] [P4]
    
    Each patch is 2x2 = 4 features.
    """
    X = X.reshape(-1, 4, 4)
    p1 = X[:, 0:2, 0:2].reshape(-1, 4)  # Top-left
    p2 = X[:, 0:2, 2:4].reshape(-1, 4)  # Top-right
    p3 = X[:, 2:4, 0:2].reshape(-1, 4)  # Bottom-left
    p4 = X[:, 2:4, 2:4].reshape(-1, 4)  # Bottom-right
    return p1, p2, p3, p4


# =============================================================================
# Quantum Feature Extractors
# =============================================================================

def create_patch_encoder(
    patch_data: np.ndarray,
    patch_name: str,
    num_qubits: int = 4,
    n_basis: int = 8,
    n_layers: int = 2,
) -> QuantumTensor:
    """Create a quantum feature extractor for a single patch.
    
    Architecture:
      - Hardware-efficient ansatz with data re-uploading
      - Multiple entanglement layers
      - ISwitch over batch and basis dimensions
    
    Note: Index names come from ISwitch parameter names.
    """
    n_batch = len(patch_data)
    
    qc = QuantumCircuit(num_qubits)
    qc.add_register(ClassicalRegister(num_qubits))
    
    # Parameter names become index names in the QuantumTensor
    batch_param = Parameter("batch")
    basis_param = Parameter(f"{patch_name}_basis")
    
    def make_encoding_circuit(idx: int) -> QuantumCircuit:
        """Expressive data encoding with re-uploading."""
        c = QuantumCircuit(num_qubits)
        x = patch_data[idx]
        
        for layer in range(n_layers):
            # Feature encoding
            for i in range(num_qubits):
                c.ry(float(x[i % len(x)]), i)
                if layer > 0:
                    c.rz(float(x[(i + 1) % len(x)]), i)
            
            # Entanglement ring
            for i in range(num_qubits):
                c.cx(i, (i + 1) % num_qubits)
            
            # Additional entanglement pattern on even layers
            if layer % 2 == 0:
                for i in range(0, num_qubits - 1, 2):
                    c.cz(i, i + 1)
        
        # Final feature interaction layer
        for i in range(num_qubits):
            prod_angle = float(x[i % len(x)] * x[(i + 1) % len(x)])
            c.rx(prod_angle, i)
        
        return c
    
    def make_basis_circuit(idx: int) -> QuantumCircuit:
        """Different measurement bases for feature diversity."""
        c = QuantumCircuit(num_qubits)
        # Rotate each qubit based on basis index
        angle = idx * np.pi / n_basis
        for i in range(num_qubits):
            qubit_angle = angle + i * np.pi / (2 * num_qubits)
            c.ry(qubit_angle, i)
        return c
    
    batch_iswitch = ISwitch(batch_param, num_qubits, n_batch, make_encoding_circuit)
    basis_iswitch = ISwitch(basis_param, num_qubits, n_basis, make_basis_circuit)
    
    qc.append(batch_iswitch, range(num_qubits))
    qc.append(basis_iswitch, range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    
    return QuantumTensor(qc)


# =============================================================================
# Quantum Feature Extraction
# =============================================================================

def extract_quantum_features(
    patches: list[np.ndarray],
    patch_names: list[str],
    n_basis: int = 8,
    n_layers: int = 2,
    backend: str = "cudaq",
) -> tuple[list[torch.Tensor], float]:
    """Extract quantum features from all patches.
    
    This is done ONCE and cached for efficient training.
    
    Returns:
        features: List of tensors, each [batch, n_basis]
        quantum_time: Time spent on quantum evaluation
    """
    features = []
    total_quantum_time = 0.0
    
    for patch_data, patch_name in zip(patches, patch_names):
        print(f"  Extracting features for {patch_name}...")
        
        # Create quantum tensor
        qtensor = create_patch_encoder(
            patch_data, patch_name, 
            n_basis=n_basis, n_layers=n_layers
        )
        
        # HEinsum to get [batch, basis] features
        heinsum = HEinsum(
            qtensors=[qtensor],
            ctensors=[],
            input_tensors=[],
            output_inds=("batch", f"{patch_name}_basis"),
        )
        
        # Execute
        runtime = HEinsumRuntime(heinsum, backend=backend, dtype=torch.float64)
        runtime.prepare(opt_kwargs={"max_repeats": 16, "progbar": False})
        
        result, timing = runtime.execute()
        features.append(result.detach())
        total_quantum_time += timing.quantum_eval_time
        
        print(f"    Shape: {result.shape}, quantum_time: {timing.quantum_eval_time*1000:.0f}ms")
    
    return features, total_quantum_time


# =============================================================================
# Classical Network on Quantum Features
# =============================================================================

class QuantumFeatureClassifier(nn.Module):
    """MLP classifier on concatenated quantum features.
    
    Architecture:
        [q1, q2, q3, q4] -> concat -> MLP -> logits
        
    Where each qi is [batch, n_basis] quantum features.
    """
    
    def __init__(self, n_patches: int, n_basis: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        
        input_dim = n_patches * n_basis
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [batch, n_basis] tensors
            
        Returns:
            logits: [batch] tensor
        """
        x = torch.cat(features, dim=-1)  # [batch, n_patches * n_basis]
        return self.network(x).squeeze(-1)


class HierarchicalQuantumClassifier(nn.Module):
    """Hierarchical classifier that respects patch spatial structure.
    
    Architecture:
        q1[batch, b] ─┬─ MLP1 ─┐
        q2[batch, b] ─┘        │
                               ├─ MLP3 ─→ logits[batch]
        q3[batch, b] ─┬─ MLP2 ─┘
        q4[batch, b] ─┘
    
    This preserves the spatial hierarchy of the image patches.
    """
    
    def __init__(self, n_basis: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        
        # Combine top patches (p1, p2)
        self.top_combine = nn.Sequential(
            nn.Linear(n_basis * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Combine bottom patches (p3, p4)
        self.bottom_combine = nn.Sequential(
            nn.Linear(n_basis * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Final combination
        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [q1, q2, q3, q4] each [batch, n_basis]
            
        Returns:
            logits: [batch]
        """
        q1, q2, q3, q4 = features
        
        # Combine spatial neighbors
        top = self.top_combine(torch.cat([q1, q2], dim=-1))
        bottom = self.bottom_combine(torch.cat([q3, q4], dim=-1))
        
        # Final combination
        combined = torch.cat([top, bottom], dim=-1)
        return self.final(combined).squeeze(-1)


class AttentionQuantumClassifier(nn.Module):
    """Attention-based classifier over quantum patch features.
    
    Treats each patch as a token and applies self-attention.
    """
    
    def __init__(self, n_basis: int, hidden_dim: int = 32, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.n_patches = 4
        
        # Project quantum features to hidden dim
        self.input_proj = nn.Linear(n_basis, hidden_dim)
        
        # Learnable position embeddings for patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, hidden_dim) * 0.02)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [q1, q2, q3, q4] each [batch, n_basis]
            
        Returns:
            logits: [batch]
        """
        # Stack patches: [batch, 4, n_basis]
        x = torch.stack(features, dim=1)
        
        # Project to hidden dim and add position embeddings
        x = self.input_proj(x) + self.pos_embed
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        # Global average pooling over patches
        x = x.mean(dim=1)  # [batch, hidden_dim]
        
        return self.classifier(x).squeeze(-1)


# =============================================================================
# Training Pipeline
# =============================================================================

def train_mnist(
    digit_a: int = 0,
    digit_b: int = 1,
    n_samples: int = 400,
    n_basis: int = 8,
    n_layers: int = 2,
    hidden_dim: int = 32,
    model_type: str = "hierarchical",  # "mlp", "hierarchical", or "attention"
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    backend: str = "cudaq",
):
    """Train a quantum-classical hybrid network on MNIST.
    
    The training pipeline:
    1. Extract quantum features ONCE (expensive quantum part)
    2. Train classical network on cached features (fast)
    """
    print("=" * 70)
    print("MNIST CLASSIFICATION WITH QUANTUM FEATURE NETWORK")
    print("=" * 70)
    print()
    
    # Load data
    X, y = load_mnist_binary(digit_a=digit_a, digit_b=digit_b, n_samples=n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nDataset: MNIST ({digit_a} vs {digit_b})")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Image size: 4x4 (downsampled)")
    print(f"Patches: 4 x (2x2)")
    print()
    
    # Extract patches
    p1_train, p2_train, p3_train, p4_train = extract_patches(X_train)
    p1_test, p2_test, p3_test, p4_test = extract_patches(X_test)
    
    # ==========================================================================
    # QUANTUM FEATURE EXTRACTION (done once)
    # ==========================================================================
    print("-" * 70)
    print("QUANTUM FEATURE EXTRACTION")
    print("-" * 70)
    print(f"Configuration: n_basis={n_basis}, n_layers={n_layers}")
    print()
    
    print("Extracting training features...")
    train_features, train_quantum_time = extract_quantum_features(
        [p1_train, p2_train, p3_train, p4_train],
        ["p1", "p2", "p3", "p4"],
        n_basis=n_basis,
        n_layers=n_layers,
        backend=backend,
    )
    
    print(f"\nExtracting test features...")
    test_features, test_quantum_time = extract_quantum_features(
        [p1_test, p2_test, p3_test, p4_test],
        ["p1", "p2", "p3", "p4"],
        n_basis=n_basis,
        n_layers=n_layers,
        backend=backend,
    )
    
    total_quantum_time = train_quantum_time + test_quantum_time
    print(f"\nTotal quantum feature extraction time: {total_quantum_time:.2f}s")
    print(f"Feature dimension per patch: {n_basis}")
    print(f"Total feature dimension: {4 * n_basis}")
    
    # ==========================================================================
    # CLASSICAL TRAINING (fast, on cached features)
    # ==========================================================================
    print()
    print("-" * 70)
    print("CLASSICAL NETWORK TRAINING")
    print("-" * 70)
    
    # Create model
    if model_type == "mlp":
        model = QuantumFeatureClassifier(
            n_patches=4,
            n_basis=n_basis,
            hidden_dims=[hidden_dim, hidden_dim],
            dropout=dropout,
        )
    elif model_type == "hierarchical":
        model = HierarchicalQuantumClassifier(
            n_basis=n_basis,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif model_type == "attention":
        model = AttentionQuantumClassifier(
            n_basis=n_basis,
            hidden_dim=hidden_dim,
            n_heads=2,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model = model.double()  # Match quantum feature dtype
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}")
    print(f"Parameters: {n_params:,}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Dropout: {dropout}")
    print()
    
    # Targets
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print("Training...")
    best_test_acc = 0.0
    best_epoch = 0
    train_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward
        logits = model(train_features)
        probs = torch.sigmoid(logits)
        
        # Loss
        loss = nn.functional.binary_cross_entropy(probs, y_train_t)
        
        # Backward
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            train_preds = (probs > 0.5).float()
            train_acc = (train_preds == y_train_t).float().mean()
            
            test_logits = model(test_features)
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs > 0.5).float()
            test_acc = (test_preds == y_test_t).float().mean()
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, "
                  f"train_acc={train_acc:.2%}, test_acc={test_acc:.2%}")
    
    train_time = time.time() - train_start
    
    # ==========================================================================
    # FINAL RESULTS
    # ==========================================================================
    print()
    print("-" * 70)
    print("FINAL RESULTS")
    print("-" * 70)
    
    model.eval()
    with torch.no_grad():
        train_logits = model(train_features)
        train_probs = torch.sigmoid(train_logits)
        train_acc = ((train_probs > 0.5).float() == y_train_t).float().mean()
        
        test_logits = model(test_features)
        test_probs = torch.sigmoid(test_logits)
        test_acc = ((test_probs > 0.5).float() == y_test_t).float().mean()
    
    print(f"\nFinal Accuracy:")
    print(f"  Train: {train_acc:.2%}")
    print(f"  Test:  {test_acc:.2%}")
    print(f"  Best test: {best_test_acc:.2%} (epoch {best_epoch})")
    
    print(f"\nTiming:")
    print(f"  Quantum feature extraction: {total_quantum_time:.2f}s")
    print(f"  Classical training ({epochs} epochs): {train_time:.2f}s")
    print(f"  Total: {total_quantum_time + train_time:.2f}s")
    
    return {
        "train_acc": train_acc.item(),
        "test_acc": test_acc.item(),
        "best_test_acc": best_test_acc.item(),
        "quantum_time": total_quantum_time,
        "train_time": train_time,
    }


def compare_architectures():
    """Compare different classifier architectures on quantum features."""
    print("=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    results = {}
    
    for model_type in ["mlp", "hierarchical", "attention"]:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'='*70}")
        
        result = train_mnist(
            digit_a=0,
            digit_b=1,
            n_samples=400,
            n_basis=8,
            n_layers=2,
            hidden_dim=32,
            model_type=model_type,
            epochs=100,
            lr=0.01,
            weight_decay=0.01,
            dropout=0.1,
        )
        results[model_type] = result
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'Train Acc':<12} {'Test Acc':<12} {'Best Test':<12}")
    print("-" * 50)
    for model_type, result in results.items():
        print(f"{model_type:<15} {result['train_acc']:.2%}       "
              f"{result['test_acc']:.2%}       {result['best_test_acc']:.2%}")


if __name__ == "__main__":
    # Single model training
    train_mnist(
        digit_a=0,
        digit_b=1,
        n_samples=400,
        n_basis=8,
        n_layers=2,
        hidden_dim=32,
        model_type="attention",  # Try: "mlp", "hierarchical", "attention"
        epochs=100,
        lr=0.01,
        weight_decay=0.01,
        dropout=0.1,
    )
    
    # Uncomment to compare all architectures:
    # compare_architectures()
