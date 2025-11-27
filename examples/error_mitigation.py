"""Error mitigation examples using HEinsum and QuantumTensor.

This module demonstrates how QTPU's tensor-based representation enables
efficient error mitigation workflows:
1. ZNE (Zero Noise Extrapolation)
2. Pauli Twirling
3. PEC (Probabilistic Error Cancellation)

The key insight: error mitigation is fundamentally a tensor network problem.
QTPU represents mitigation as tensor contractions, avoiding the exponential
enumeration that naive approaches require.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RZGate, RXGate, RYGate, IGate, XGate, YGate, ZGate

from qtpu.tensor import ISwitch, QuantumTensor, CTensor, TensorSpec
from qtpu.heinsum import HEinsum


# =============================================================================
# ZNE (Zero Noise Extrapolation)
# =============================================================================

def create_zne_iswitch(
    qc: QuantumCircuit,
    qubit: int,
    gate: QuantumCircuit,
    noise_levels: list[int] = [1, 3, 5],
    idx: str = "zne",
) -> ISwitch:
    """Create an ISwitch for ZNE with different noise amplification levels.
    
    Noise is amplified by folding: G -> G G^dag G (for 3x) -> G G^dag G G^dag G (for 5x)
    
    Args:
        qc: The quantum circuit (for context).
        qubit: The qubit index.
        gate: The gate to fold (as a small circuit).
        noise_levels: List of noise scale factors (must be odd integers).
        idx: Index name for the tensor dimension.
    
    Returns:
        ISwitch instruction encoding the noise levels.
    """
    param = Parameter(idx)
    num_qubits = gate.num_qubits
    
    def selector(level_idx: int) -> QuantumCircuit:
        level = noise_levels[level_idx]
        folded = QuantumCircuit(num_qubits)
        # Fold: G (G^dag G)^((level-1)/2)
        folded.compose(gate, inplace=True)
        for _ in range((level - 1) // 2):
            folded.compose(gate.inverse(), inplace=True)
            folded.compose(gate, inplace=True)
        return folded
    
    return ISwitch(param, num_qubits, len(noise_levels), selector)


def zne_coefficients(noise_levels: list[int]) -> np.ndarray:
    """Richardson extrapolation coefficients for ZNE.
    
    For noise levels [1, 3, 5], we fit a polynomial and extrapolate to 0.
    """
    n = len(noise_levels)
    # Vandermonde matrix for polynomial fit
    V = np.vander(noise_levels, increasing=True)
    # We want coefficients c such that sum(c_i * f(level_i)) = f(0)
    # This is the first row of V^{-1}
    coeffs = np.linalg.inv(V)[0]
    return coeffs


def create_zne_heinsum(circuit: QuantumCircuit, num_noisy_gates: int = 3) -> HEinsum:
    """Create a HEinsum for ZNE on a circuit.
    
    Example: 3 noisy gates with 3 noise levels each.
    Naive: 3^3 = 27 circuit executions
    QTPU: contracts efficiently, potentially fewer unique circuits needed.
    
    Args:
        circuit: Base quantum circuit.
        num_noisy_gates: Number of gates to apply ZNE to.
    
    Returns:
        HEinsum specification for ZNE.
    """
    noise_levels = [1, 3, 5]
    
    # Create a circuit with ISwitch for each noisy gate
    zne_circuit = QuantumCircuit(circuit.num_qubits)
    
    # For simplicity, apply ZNE to first num_noisy_gates single-qubit gates
    gate_count = 0
    for instr in circuit:
        if gate_count < num_noisy_gates and instr.operation.num_qubits == 1:
            # Create folded gate options
            base_gate = QuantumCircuit(1)
            base_gate.append(instr.operation, [0])
            
            iswitch = create_zne_iswitch(
                zne_circuit, 
                instr.qubits[0].index if hasattr(instr.qubits[0], 'index') else 0,
                base_gate,
                noise_levels,
                idx=f"zne_{gate_count}"
            )
            zne_circuit.append(iswitch, instr.qubits)
            gate_count += 1
        else:
            zne_circuit.append(instr.operation, instr.qubits, instr.clbits)
    
    qtensor = QuantumTensor(zne_circuit)
    
    # ZNE coefficients tensor - contracts over all ZNE indices
    coeffs = zne_coefficients(noise_levels)
    ctensors = []
    for i in range(num_noisy_gates):
        ctensor = CTensor(coeffs, (f"zne_{i}",))
        ctensors.append(ctensor)
    
    return HEinsum(
        qtensors=[qtensor],
        ctensors=ctensors,
        input_tensors=[],
        output_inds=(),  # Scalar output (expectation value)
    )


# =============================================================================
# Pauli Twirling
# =============================================================================

PAULIS = [IGate(), XGate(), YGate(), ZGate()]
PAULI_NAMES = ['I', 'X', 'Y', 'Z']


def create_twirl_iswitch(idx: str) -> ISwitch:
    """Create an ISwitch for Pauli twirling on a single qubit.
    
    Twirling inserts random Paulis P before and P^dag after a gate,
    which averages coherent errors to stochastic errors.
    
    Args:
        idx: Index name for the tensor dimension.
    
    Returns:
        ISwitch with 4 options (I, X, Y, Z).
    """
    param = Parameter(idx)
    
    def selector(pauli_idx: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.append(PAULIS[pauli_idx], [0])
        return qc
    
    return ISwitch(param, 1, 4, selector)


def create_twirl_heinsum(
    circuit: QuantumCircuit, 
    num_twirl_locations: int = 4
) -> HEinsum:
    """Create a HEinsum for Pauli twirling.
    
    Twirling averages over random Pauli frames. The coefficient tensor
    is uniform (1/4 for each Pauli).
    
    Naive: 4^num_locations circuit executions
    QTPU: Can sample subset and still get valid estimate
    
    Args:
        circuit: Base quantum circuit.
        num_twirl_locations: Number of locations to insert twirls.
    
    Returns:
        HEinsum specification for twirling.
    """
    twirl_circuit = QuantumCircuit(circuit.num_qubits)
    
    # Insert twirl before each 2Q gate (simplified: just insert at start)
    for i in range(min(num_twirl_locations, circuit.num_qubits)):
        iswitch = create_twirl_iswitch(f"twirl_{i}")
        twirl_circuit.append(iswitch, [i])
    
    # Add original circuit
    twirl_circuit.compose(circuit, inplace=True)
    
    # Insert inverse twirl at end
    for i in range(min(num_twirl_locations, circuit.num_qubits)):
        iswitch = create_twirl_iswitch(f"twirl_inv_{i}")
        twirl_circuit.append(iswitch, [i])
    
    qtensor = QuantumTensor(twirl_circuit)
    
    # Uniform averaging coefficients
    # For proper twirling, twirl and twirl_inv must match (P ... P^dag)
    # We encode this with a delta tensor that contracts matching indices
    ctensors = []
    for i in range(min(num_twirl_locations, circuit.num_qubits)):
        # Delta tensor: shape (4, 4), only diagonal is 1/4
        delta = np.eye(4) / 4
        ctensor = CTensor(delta, (f"twirl_{i}", f"twirl_inv_{i}"))
        ctensors.append(ctensor)
    
    return HEinsum(
        qtensors=[qtensor],
        ctensors=ctensors,
        input_tensors=[],
        output_inds=(),
    )


# =============================================================================
# PEC (Probabilistic Error Cancellation)
# =============================================================================

def depolarizing_qpd_coefficients() -> tuple[np.ndarray, list]:
    """QPD coefficients for depolarizing noise on a single qubit.
    
    A noisy gate G_noisy can be decomposed as:
    G_ideal = sum_i c_i * B_i
    where B_i are basis operations and c_i are quasi-probabilities (can be negative).
    
    For depolarizing noise with error rate p:
    G_noisy = (1-p) G_ideal + p/3 (X G_ideal X + Y G_ideal Y + Z G_ideal Z)
    
    Returns:
        Tuple of (coefficients, basis_operations).
    """
    # Simplified: assume we know the noise model
    # In practice, this comes from gate set tomography
    p = 0.01  # 1% depolarizing error rate
    
    # QPD: G_ideal = (1 + 3p/(1-p)) G_noisy - p/(1-p) * (X G X + Y G Y + Z G Z)
    gamma = p / (1 - p)
    coeffs = np.array([1 + 3*gamma, -gamma, -gamma, -gamma])
    
    return coeffs, PAULIS


def create_pec_iswitch(gate: QuantumCircuit, idx: str) -> ISwitch:
    """Create an ISwitch for PEC basis operations.
    
    Args:
        gate: The ideal gate to error-cancel.
        idx: Index name for the tensor dimension.
    
    Returns:
        ISwitch with 4 PEC basis options.
    """
    param = Parameter(idx)
    _, basis_ops = depolarizing_qpd_coefficients()
    
    def selector(basis_idx: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        # Apply Pauli before and after (simplified PEC basis)
        qc.append(basis_ops[basis_idx], [0])
        qc.compose(gate, inplace=True)
        qc.append(basis_ops[basis_idx], [0])  # Pauli^2 = I for X,Y,Z
        return qc
    
    return ISwitch(param, 1, 4, selector)


def create_pec_heinsum(
    circuit: QuantumCircuit,
    num_pec_gates: int = 3
) -> HEinsum:
    """Create a HEinsum for PEC.
    
    PEC decomposes noisy gates into quasi-probability distributions.
    This is where the 6^k overhead comes from in wire cutting (special case).
    
    Naive: 4^num_pec_gates executions with weighted combination
    QTPU: Tensor contraction finds optimal execution strategy
    
    Args:
        circuit: Base quantum circuit.
        num_pec_gates: Number of gates to apply PEC to.
    
    Returns:
        HEinsum specification for PEC.
    """
    pec_circuit = QuantumCircuit(circuit.num_qubits)
    
    # Apply PEC to first num_pec_gates single-qubit gates
    gate_count = 0
    for instr in circuit:
        if gate_count < num_pec_gates and instr.operation.num_qubits == 1:
            base_gate = QuantumCircuit(1)
            base_gate.append(instr.operation, [0])
            
            iswitch = create_pec_iswitch(base_gate, f"pec_{gate_count}")
            pec_circuit.append(iswitch, instr.qubits)
            gate_count += 1
        else:
            pec_circuit.append(instr.operation, instr.qubits, instr.clbits)
    
    qtensor = QuantumTensor(pec_circuit)
    
    # PEC quasi-probability coefficients
    coeffs, _ = depolarizing_qpd_coefficients()
    ctensors = []
    for i in range(num_pec_gates):
        ctensor = CTensor(coeffs, (f"pec_{i}",))
        ctensors.append(ctensor)
    
    return HEinsum(
        qtensors=[qtensor],
        ctensors=ctensors,
        input_tensors=[],
        output_inds=(),
    )


# =============================================================================
# Combined Error Mitigation
# =============================================================================

def create_combined_heinsum(
    circuit: QuantumCircuit,
    num_zne: int = 2,
    num_twirl: int = 2,
    num_pec: int = 2,
) -> HEinsum:
    """Create a HEinsum combining ZNE + Twirling + PEC.
    
    This demonstrates the power of tensor-based error mitigation:
    - ZNE indices: shape (3,) each -> 3^num_zne
    - Twirl indices: shape (4,4) each -> 4^(2*num_twirl) but delta reduces
    - PEC indices: shape (4,) each -> 4^num_pec
    
    Naive total: 3^2 * 4^4 * 4^2 = 9 * 256 * 16 = 36,864 circuits!
    QTPU: Contracts efficiently, many redundant computations eliminated.
    
    Args:
        circuit: Base quantum circuit.
        num_zne: Number of gates for ZNE.
        num_twirl: Number of qubits for twirling.
        num_pec: Number of gates for PEC.
    
    Returns:
        HEinsum specification for combined mitigation.
    """
    noise_levels = [1, 3, 5]
    
    combined_circuit = QuantumCircuit(circuit.num_qubits)
    ctensors = []
    
    # Phase 1: Add twirl at start
    for i in range(min(num_twirl, circuit.num_qubits)):
        iswitch = create_twirl_iswitch(f"twirl_pre_{i}")
        combined_circuit.append(iswitch, [i])
    
    # Phase 2: Add circuit gates with ZNE and PEC
    zne_count = 0
    pec_count = 0
    
    for instr in circuit:
        if instr.operation.num_qubits == 1:
            base_gate = QuantumCircuit(1)
            base_gate.append(instr.operation, [0])
            
            if zne_count < num_zne:
                # Apply ZNE
                iswitch = create_zne_iswitch(
                    combined_circuit, 0, base_gate, noise_levels, f"zne_{zne_count}"
                )
                combined_circuit.append(iswitch, instr.qubits)
                ctensors.append(CTensor(zne_coefficients(noise_levels), (f"zne_{zne_count}",)))
                zne_count += 1
            elif pec_count < num_pec:
                # Apply PEC
                iswitch = create_pec_iswitch(base_gate, f"pec_{pec_count}")
                combined_circuit.append(iswitch, instr.qubits)
                coeffs, _ = depolarizing_qpd_coefficients()
                ctensors.append(CTensor(coeffs, (f"pec_{pec_count}",)))
                pec_count += 1
            else:
                combined_circuit.append(instr.operation, instr.qubits, instr.clbits)
        else:
            combined_circuit.append(instr.operation, instr.qubits, instr.clbits)
    
    # Phase 3: Add inverse twirl at end
    for i in range(min(num_twirl, circuit.num_qubits)):
        iswitch = create_twirl_iswitch(f"twirl_post_{i}")
        combined_circuit.append(iswitch, [i])
        # Delta tensor for twirl matching
        delta = np.eye(4) / 4
        ctensors.append(CTensor(delta, (f"twirl_pre_{i}", f"twirl_post_{i}")))
    
    qtensor = QuantumTensor(combined_circuit)
    
    return HEinsum(
        qtensors=[qtensor],
        ctensors=ctensors,
        input_tensors=[],
        output_inds=(),
    )


# =============================================================================
# Demonstration: QTPU Advantage
# =============================================================================

def compare_approaches():
    """Compare naive enumeration vs QTPU tensor contraction."""
    from qiskit.circuit.library import efficient_su2
    
    print("=" * 60)
    print("Error Mitigation: Naive vs QTPU Comparison")
    print("=" * 60)
    
    # Create a simple variational circuit
    circuit = efficient_su2(4, reps=1).decompose()
    
    # Test different mitigation strategies
    strategies = [
        ("ZNE only (3 gates)", lambda: create_zne_heinsum(circuit, 3)),
        ("Twirl only (4 qubits)", lambda: create_twirl_heinsum(circuit, 4)),
        ("PEC only (3 gates)", lambda: create_pec_heinsum(circuit, 3)),
        ("Combined (2+2+2)", lambda: create_combined_heinsum(circuit, 2, 2, 2)),
    ]
    
    for name, create_fn in strategies:
        print(f"\n{name}:")
        print("-" * 40)
        
        heinsum = create_fn()
        qtensor = heinsum.quantum_tensors[0]
        
        # Naive approach: enumerate all circuits
        naive_circuits = np.prod(qtensor.shape) if qtensor.shape else 1
        
        print(f"  Tensor shape: {qtensor.shape}")
        print(f"  Tensor indices: {qtensor.inds}")
        print(f"  Naive circuits needed: {naive_circuits:,}")
        
        # QTPU approach: use tensor contraction
        tree, arrays = heinsum.to_dummy_tn()
        if tree is not None:
            flops = tree.contraction_cost()
            print(f"  Contraction FLOPs: {flops:,.0f}")
            print(f"  Speedup potential: {naive_circuits / max(1, flops / 1000):.1f}x")
        else:
            print(f"  Single tensor (no contraction needed)")
        
        print(f"  Einsum expression: {heinsum.einsum_expr}")


def demonstrate_scaling():
    """Show how QTPU scales better than naive for combined mitigation."""
    from qiskit.circuit.library import efficient_su2
    
    print("\n" + "=" * 60)
    print("Scaling Analysis: Combined Error Mitigation")
    print("=" * 60)
    
    print(f"\n{'Gates':<10} {'Naive Circuits':<20} {'QTPU FLOPs':<20} {'Ratio':<15}")
    print("-" * 65)
    
    for num_gates in [2, 4, 6, 8]:
        circuit = efficient_su2(num_gates, reps=1).decompose()
        
        # Combined mitigation
        heinsum = create_combined_heinsum(
            circuit, 
            num_zne=num_gates // 2,
            num_twirl=min(num_gates, 4),
            num_pec=num_gates // 2
        )
        
        qtensor = heinsum.quantum_tensors[0]
        naive = np.prod(qtensor.shape) if qtensor.shape else 1
        
        tree, _ = heinsum.to_dummy_tn()
        flops = tree.contraction_cost() if tree else 1
        
        ratio = naive / max(1, flops / 1000)
        print(f"{num_gates:<10} {naive:<20,} {flops:<20,.0f} {ratio:<15.1f}x")


if __name__ == "__main__":
    compare_approaches()
    demonstrate_scaling()
