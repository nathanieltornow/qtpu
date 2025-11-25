"""
Hybrid Tensor Network (hTN) Use Cases

Demonstrates hTN as a unified abstraction for quantum-classical compute.
The key insight: tensors can be quantum (CircuitTensor) or classical (qtn.Tensor),
and contraction is expressed via tensor network indices.

Uses QTPU's existing infrastructure:
- CircuitTensor: quantum tensors parameterized by ISwitch
- HybridTensorNetwork: combines quantum and classical tensors  
- Evaluators: ExpvalEvaluator, SamplerEvaluator, etc.
- contract/evaluate: execute the hTN

Use Cases:
1. VQE - Variational Quantum Eigensolver
2. Quantum Kernel Methods  
3. Quantum Error Mitigation
4. Distributed Quantum Computing (Circuit Cutting) - QTPU's core!
5. Hybrid Quantum-Classical ML
"""

import numpy as np
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qtpu.tensor import QuantumTensor, HybridTensorNetwork, ISwitch
from qtpu.evaluators import ExpvalEvaluator
from qtpu.contract import contract, evaluate


# -----------------------------------------------------------------------------
# Use Case 1: VQE (Variational Quantum Eigensolver)
# -----------------------------------------------------------------------------


def use_case_vqe():
    """
    VQE: Find ground state energy of a Hamiltonian.
    
    hTN structure:
    - CircuitTensor: ansatz with ISwitch selecting different parameter values
    - Classical tensor: Hamiltonian coefficients
    - Contraction: computes weighted sum of expectation values
    """
    print("=" * 60)
    print("Use Case 1: VQE (Variational Quantum Eigensolver)")
    print("=" * 60)

    # Simple Hamiltonian: H = c0*Z + c1*X
    # We discretize parameter space and use ISwitch to select
    
    # Create ansatz with ISwitch for parameter selection
    idx = Parameter("theta_idx")
    
    # Discretize rotation angles
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    def angle_selector(i: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(angles[i], 0)
        return qc
    
    iswitch = ISwitch(idx, num_qubits=1, size=len(angles), selector=angle_selector)
    
    ansatz = QuantumCircuit(1, 1)
    ansatz.append(iswitch, [0])
    ansatz.measure(0, 0)
    
    # Create CircuitTensor from ansatz
    circuit_tensor = QuantumTensor(ansatz)
    print(f"\nCircuitTensor shape: {circuit_tensor.shape}")
    print(f"CircuitTensor indices: {circuit_tensor.inds}")
    
    # Hamiltonian coefficients (classical tensor)
    # For each angle, we have a weight
    coefficients = qtn.Tensor(
        np.array([0.5, 0.3, 0.1, 0.3, 0.5]),  # Example weights
        inds=circuit_tensor.inds
    )
    
    # Create hTN
    htn = HybridTensorNetwork(
        qtensors=[circuit_tensor],
        ctensors=[coefficients]
    )
    
    print(f"\nhTN: {len(htn.qtensors)} quantum tensor(s), {len(htn.ctensors)} classical tensor(s)")
    
    # Evaluate and contract
    evaluator = ExpvalEvaluator()
    result = contract(htn, evaluator)
    
    # Result is scalar when fully contracted
    result_val = float(result) if np.isscalar(result) else float(result.data)
    print(f"VQE Energy (weighted sum): {result_val:.4f}")
    print("\nThe contraction sums circuit outputs weighted by Hamiltonian coefficients.")


# -----------------------------------------------------------------------------
# Use Case 2: Quantum Kernel Methods
# -----------------------------------------------------------------------------


def use_case_quantum_kernel():
    """
    Quantum Kernel: Build kernel matrix K(x_i, x_j) using circuit tensors.
    
    hTN structure:
    - CircuitTensor: kernel circuit with ISwitch selecting data points
    - Classical tensor: combines kernel outputs for regression/classification
    """
    print("\n" + "=" * 60)
    print("Use Case 2: Quantum Kernel Methods")
    print("=" * 60)

    # Data points encoded as rotation angles
    X_data = [0.1, 0.5, 1.0, 1.5]
    n_points = len(X_data)
    
    # Create kernel circuit with two ISwitch (for x_i and x_j)
    idx_i = Parameter("i")
    idx_j = Parameter("j")
    
    def feature_map_i(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(X_data[k], 0)
        qc.rz(X_data[k] * 2, 0)
        return qc
    
    def feature_map_j(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        # Inverse feature map for fidelity
        qc.rz(-X_data[k] * 2, 0)
        qc.ry(-X_data[k], 0)
        return qc
    
    iswitch_i = ISwitch(idx_i, num_qubits=1, size=n_points, selector=feature_map_i)
    iswitch_j = ISwitch(idx_j, num_qubits=1, size=n_points, selector=feature_map_j)
    
    # Kernel circuit: |⟨φ(x_i)|φ(x_j)⟩|² via swap test approximation
    kernel_circuit = QuantumCircuit(1, 1)
    kernel_circuit.append(iswitch_i, [0])
    kernel_circuit.append(iswitch_j, [0])
    kernel_circuit.measure(0, 0)
    
    circuit_tensor = QuantumTensor(kernel_circuit)
    print(f"\nKernel CircuitTensor shape: {circuit_tensor.shape}")
    print(f"Indices: {circuit_tensor.inds} (data point pairs)")
    
    # Labels for kernel regression (classical tensor)
    y_labels = np.array([1.0, 1.0, -1.0, -1.0])
    label_tensor = qtn.Tensor(y_labels, inds=["i"])
    
    # For kernel regression: α = K^(-1) y, but simplified here
    # We contract kernel with labels
    htn = HybridTensorNetwork(
        qtensors=[circuit_tensor],
        ctensors=[label_tensor]
    )
    
    # Evaluate kernel matrix
    evaluator = ExpvalEvaluator()
    evaluated_tn = evaluate(htn, evaluator)
    
    print(f"\nKernel matrix shape from contraction:")
    for t in evaluated_tn.tensors:
        print(f"  Tensor with inds {t.inds}, shape {t.shape}")
    
    result = evaluated_tn.contract(all, output_inds=["j"])
    print(f"\nKernel-weighted output shape: {result.shape}")
    print(f"Values: {result.data}")


# -----------------------------------------------------------------------------
# Use Case 3: Quantum Error Mitigation  
# -----------------------------------------------------------------------------


def use_case_error_mitigation():
    """
    Error Mitigation: Combine noisy circuit outputs with mitigation tensors.
    
    hTN structure:
    - CircuitTensor: noisy circuit execution
    - Classical tensor: error mitigation matrix (quasi-probability)
    - Contraction: mitigated_result = M @ noisy_result
    """
    print("\n" + "=" * 60)
    print("Use Case 3: Quantum Error Mitigation")
    print("=" * 60)

    # Circuit with ISwitch representing different noise realizations
    # (In practice, these would be different Pauli twirling instances)
    idx = Parameter("noise_idx")
    
    def noisy_circuit_variant(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.h(0)
        # Different "noise" realizations
        if k == 1:
            qc.x(0)
            qc.x(0)  # Identity but triggers different path
        elif k == 2:
            qc.z(0)
            qc.z(0)
        return qc
    
    iswitch = ISwitch(idx, num_qubits=1, size=3, selector=noisy_circuit_variant)
    
    circuit = QuantumCircuit(1, 1)
    circuit.append(iswitch, [0])
    circuit.measure(0, 0)
    
    circuit_tensor = QuantumTensor(circuit)
    
    # Mitigation coefficients (quasi-probability decomposition)
    # These would come from noise characterization
    mitigation_coeffs = qtn.Tensor(
        np.array([1.2, -0.1, -0.1]),  # Quasi-probabilities can be negative!
        inds=circuit_tensor.inds
    )
    
    htn = HybridTensorNetwork(
        qtensors=[circuit_tensor],
        ctensors=[mitigation_coeffs]
    )
    
    print(f"\nCircuitTensor for noise variants: {circuit_tensor.shape}")
    print(f"Mitigation coefficients: {mitigation_coeffs.data}")
    print("Note: coefficients can be negative (quasi-probability)")
    
    evaluator = ExpvalEvaluator()
    result = contract(htn, evaluator)
    
    result_val = float(result) if np.isscalar(result) else float(result.data)
    print(f"\nMitigated expectation value: {result_val:.4f}")
    print("Contraction applies quasi-probability weights to circuit outputs.")


# -----------------------------------------------------------------------------
# Use Case 4: Distributed Quantum Computing (Circuit Cutting)
# -----------------------------------------------------------------------------


def use_case_circuit_cutting():
    """
    Circuit Cutting: QTPU's core use case!
    
    Large circuit → cut into subcircuits → hTN with:
    - CircuitTensors: subcircuits (fragments)
    - Classical tensors: QPD coefficients from cuts
    - Contraction: reconstructs full circuit output
    
    This is what QTPU automates via circuit_to_hybrid_tn()!
    """
    print("\n" + "=" * 60)
    print("Use Case 4: Distributed Quantum Computing (Circuit Cutting)")
    print("=" * 60)
    print(">>> This is QTPU's CORE contribution! <<<")

    from qtpu import circuit_to_hybrid_tn, cut
    
    # Create a circuit too large for a single QPU
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)  # This gate will be cut
    circuit.cx(2, 3)
    circuit.measure_all()
    
    print(f"\nOriginal circuit: {circuit.num_qubits} qubits")
    print(circuit.draw())
    
    # Cut the circuit (QTPU finds optimal cuts)
    # max_qubits constrains subcircuit width
    cut_circuit = cut(circuit, max_qubits=2)
    
    # Convert to hTN
    htn = circuit_to_hybrid_tn(cut_circuit)
    
    print(f"\nAfter cutting:")
    print(f"  Quantum tensors (subcircuits): {len(htn.qtensors)}")
    print(f"  Classical tensors (QPD coeffs): {len(htn.ctensors)}")
    
    for i, qt in enumerate(htn.qtensors):
        print(f"\n  Subcircuit {i}:")
        print(f"    Shape: {qt.shape}")
        print(f"    Indices: {qt.inds}")
        print(f"    Qubits: {qt.circuit.num_qubits}")
    
    for i, ct in enumerate(htn.ctensors):
        print(f"\n  Classical tensor {i}:")
        print(f"    Shape: {ct.shape}")
        print(f"    Indices: {ct.inds}")
    
    # Contract to get result
    evaluator = ExpvalEvaluator()
    result = contract(htn, evaluator)
    
    print(f"\nContracted result: {result}")
    print("\nQTPU handles: cutting → subcircuit generation → QPD tensors → contraction")


# -----------------------------------------------------------------------------
# Use Case 5: Hybrid Quantum-Classical ML
# -----------------------------------------------------------------------------


def use_case_hybrid_ml():
    """
    Hybrid ML: Quantum feature extraction + classical processing.
    
    hTN structure:
    - CircuitTensor: quantum feature map with ISwitch for batch
    - Classical tensor: weight matrix, embeddings
    - Contraction: forward pass of hybrid model
    """
    print("\n" + "=" * 60)
    print("Use Case 5: Hybrid Quantum-Classical ML")
    print("=" * 60)

    # Batch of data points (ISwitch selects which one)
    batch_data = [0.1, 0.5, 1.0, 1.5, 2.0]
    batch_size = len(batch_data)
    
    batch_idx = Parameter("batch")
    
    def data_encoder(i: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(batch_data[i], 0)
        qc.rz(batch_data[i] * 0.5, 0)
        return qc
    
    iswitch = ISwitch(batch_idx, num_qubits=1, size=batch_size, selector=data_encoder)
    
    # Quantum feature circuit
    feature_circuit = QuantumCircuit(1, 1)
    feature_circuit.append(iswitch, [0])
    feature_circuit.h(0)  # Additional processing
    feature_circuit.measure(0, 0)
    
    circuit_tensor = QuantumTensor(feature_circuit)
    
    # Classical weight tensor (learned parameters in real ML)
    # Maps quantum features to output
    weights = qtn.Tensor(
        np.array([0.8, 0.6, 0.2, -0.3, -0.7]),  # One weight per batch element
        inds=["batch"]
    )
    
    htn = HybridTensorNetwork(
        qtensors=[circuit_tensor],
        ctensors=[weights]
    )
    
    print(f"\nQuantum feature tensor: {circuit_tensor.shape}")
    print(f"Classical weights: {weights.shape}")
    
    evaluator = ExpvalEvaluator()
    result = contract(htn, evaluator)
    
    result_val = float(result) if np.isscalar(result) else float(result.data)
    print(f"\nHybrid ML output: {result_val:.4f}")
    print("This is quantum_features · weights (dot product)")
    
    # For a full model, you'd have multiple contractions with different
    # classical tensors (weight matrices, biases, etc.)


# -----------------------------------------------------------------------------
# Use Case 6: Multi-Index Contraction (Complex hTN)
# -----------------------------------------------------------------------------


def use_case_complex_htn():
    """
    Complex hTN: Multiple quantum and classical tensors with shared indices.
    
    Demonstrates the full power of tensor network contraction.
    """
    print("\n" + "=" * 60)
    print("Use Case 6: Complex hTN with Multiple Tensors")
    print("=" * 60)

    # Two quantum circuits with shared index structure
    idx_a = Parameter("a")
    idx_b = Parameter("b")
    
    def circuit_a_selector(i: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(i * np.pi / 4, 0)
        return qc
    
    def circuit_b_selector(i: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.rx(i * np.pi / 3, 0)
        return qc
    
    iswitch_a = ISwitch(idx_a, num_qubits=1, size=4, selector=circuit_a_selector)
    iswitch_b = ISwitch(idx_b, num_qubits=1, size=3, selector=circuit_b_selector)
    
    # Circuit 1: indexed by 'a'
    circuit1 = QuantumCircuit(1, 1)
    circuit1.append(iswitch_a, [0])
    circuit1.measure(0, 0)
    
    # Circuit 2: indexed by 'b'  
    circuit2 = QuantumCircuit(1, 1)
    circuit2.append(iswitch_b, [0])
    circuit2.measure(0, 0)
    
    qt1 = QuantumTensor(circuit1)
    qt2 = QuantumTensor(circuit2)
    
    # Classical tensor connecting the two quantum tensors
    # Shape [4, 3] connects index 'a' to index 'b'
    connection_matrix = qtn.Tensor(
        np.random.randn(4, 3),
        inds=["a", "b"]
    )
    
    htn = HybridTensorNetwork(
        qtensors=[qt1, qt2],
        ctensors=[connection_matrix]
    )
    
    print(f"\nQuantum tensor 1: shape {qt1.shape}, inds {qt1.inds}")
    print(f"Quantum tensor 2: shape {qt2.shape}, inds {qt2.inds}")
    print(f"Classical tensor: shape {connection_matrix.shape}, inds {connection_matrix.inds}")
    
    # The contraction: sum over all indices
    # Result = Σ_a,b Q1[a] * C[a,b] * Q2[b]
    
    evaluator = ExpvalEvaluator()
    result = contract(htn, evaluator)
    
    result_val = float(result) if np.isscalar(result) else float(result.data)
    print(f"\nContracted result: {result_val:.4f}")
    print("Contraction: Σ_a,b circuit1[a] × connection[a,b] × circuit2[b]")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    """Run all hTN use case demonstrations."""
    print("\n" + "=" * 60)
    print("HYBRID TENSOR NETWORKS (hTN)")
    print("A Unified Abstraction for Quantum-Classical Compute")
    print("=" * 60)
    print("""
QTPU's Core Abstraction:
- CircuitTensor: quantum tensor from circuit with ISwitch
- qtn.Tensor: classical tensor (numpy array with indices)
- HybridTensorNetwork: combines both types
- contract(): evaluates and contracts the full network

The abstraction is AGNOSTIC to tensor source.
QTPU compiles and optimizes hTN execution.
    """)

    use_case_vqe()
    use_case_quantum_kernel()
    use_case_error_mitigation()
    use_case_circuit_cutting()
    use_case_hybrid_ml()
    use_case_complex_htn()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
hTN provides a UNIFIED abstraction across all use cases:

┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Use Case            │ Quantum Tensor      │ Classical Tensor    │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ VQE                 │ Ansatz circuits     │ Hamiltonian coeffs  │
│ Quantum Kernel      │ Kernel circuits     │ Labels, regression  │
│ Error Mitigation    │ Noisy variants      │ QPD coefficients    │
│ Circuit Cutting     │ Subcircuits         │ Cut tensors (QPD)   │
│ Hybrid ML           │ Feature map         │ Weights, embeddings │
│ Complex Networks    │ Multiple circuits   │ Connection matrices │
└─────────────────────┴─────────────────────┴─────────────────────┘

QTPU's role:
1. Accept hTN specification (CircuitTensors + classical Tensors)
2. Analyze and optimize (circuit cutting, scheduling)
3. Execute on available quantum/classical resources
4. Contract and return result

USER writes: HybridTensorNetwork(qtensors, ctensors)
QTPU handles: compilation, cutting, execution, contraction
    """)


if __name__ == "__main__":
    main()
