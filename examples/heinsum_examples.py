"""
Examples demonstrating the HEinsum API.

HEinsum separates STRUCTURE from DATA:
- HybridTensorNetwork defines the tensor network topology
- Classical tensors are TensorSpecs (indices + shape, no data)
- At call time: provide params + actual classical tensor data

This is the natural interface for QTPU's circuit cutting:
- cut() + circuit_to_hybrid_tn() gives you the structure
- heinsum() compiles it to a callable
- Call with different QPD coefficients / parameter values
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qtpu.tensor import QuantumTensor, HybridTensorNetwork, ISwitch
from qtpu.heinsum import heinsum, TensorSpec, compile_htn
from qtpu.evaluators import ExpvalEvaluator
import quimb.tensor as qtn


def example_basic():
    """Basic: two circuit tensors + QPD coefficients."""
    print("=" * 60)
    print("Example 1: Basic Circuit Cutting Structure")
    print("=" * 60)
    
    # Create two subcircuits with shared index
    idx = Parameter("i")
    
    def sub_a(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(k * np.pi / 3, 0)
        return qc
    
    def sub_b(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.rx(k * np.pi / 3, 0)
        return qc
    
    iswitch = ISwitch(idx, num_qubits=1, size=6, selector=sub_a)
    circuit_a = QuantumCircuit(1, 1)
    circuit_a.append(iswitch, [0])
    circuit_a.measure(0, 0)
    
    iswitch_b = ISwitch(idx, num_qubits=1, size=6, selector=sub_b)
    circuit_b = QuantumCircuit(1, 1)
    circuit_b.append(iswitch_b, [0])
    circuit_b.measure(0, 0)
    
    # Create circuit tensors
    qt_a = QuantumTensor(circuit_a)
    qt_b = QuantumTensor(circuit_b)
    
    print(f"Circuit A: shape {qt_a.shape}, inds {qt_a.inds}")
    print(f"Circuit B: shape {qt_b.shape}, inds {qt_b.inds}")
    
    # QPD coefficients tensor - STRUCTURE ONLY (no data yet)
    # Use dummy quimb tensor with zeros to define structure
    qpd_spec = qtn.Tensor(np.zeros(6), inds=("i",))
    
    # Create HybridTensorNetwork (defines structure)
    htn = HybridTensorNetwork(
        qtensors=[qt_a, qt_b],
        ctensors=[qpd_spec],
    )
    
    print(f"\nHybridTensorNetwork:")
    print(f"  Quantum tensors: {len(htn.qtensors)}")
    print(f"  Classical tensors: {len(htn.ctensors)}")
    
    # Compile to callable
    evaluator = ExpvalEvaluator()
    expr = heinsum(htn, evaluator)
    
    print(f"\nCompiled expression: {expr}")
    print(f"Signature: {expr.signature()}")
    print(f"Expected ctensor count: {expr.num_ctensors}")
    
    # Execute with actual data
    qpd_coeffs = np.array([0.5, 0.5, 0.25, 0.25, -0.25, -0.25])
    
    result = expr(ctensor_data=[qpd_coeffs])
    
    print(f"\nResult: {result}")
    
    # Execute with different QPD coefficients
    qpd_coeffs_2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result_2 = expr(ctensor_data=[qpd_coeffs_2])
    
    print(f"Result with different coeffs: {result_2}")


def example_parameterized():
    """Parameterized circuits - bind params at call time."""
    print("\n" + "=" * 60)
    print("Example 2: Parameterized Circuits")
    print("=" * 60)
    
    # Circuit with both ISwitch (tensor index) and learnable param
    idx = Parameter("i")
    theta = Parameter("theta")  # Learnable parameter
    
    def variational_ansatz(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(k * np.pi / 4, 0)  # ISwitch selects this
        return qc
    
    iswitch = ISwitch(idx, num_qubits=1, size=4, selector=variational_ansatz)
    
    circuit = QuantumCircuit(1, 1)
    circuit.append(iswitch, [0])
    circuit.ry(theta, 0)  # Learnable Y-rotation (affects Z expectation)
    circuit.measure(0, 0)
    
    qt = QuantumTensor(circuit)
    print(f"Circuit tensor: shape {qt.shape}, inds {qt.inds}")
    print(f"Circuit params: {[p.name for p in circuit.parameters]}")
    
    # Classical coefficients (structure)
    coeffs_spec = qtn.Tensor(np.zeros(4), inds=("i",))
    
    htn = HybridTensorNetwork(qtensors=[qt], ctensors=[coeffs_spec])
    
    evaluator = ExpvalEvaluator()
    expr = heinsum(htn, evaluator)
    
    print(f"\nExpression: {expr.signature()}")
    print(f"Circuit params to bind: {expr.circuit_params}")
    
    # Execute with different theta values - theta affects the result!
    coeffs = np.array([0.25, 0.25, 0.25, 0.25])
    
    for theta_val in [0, np.pi/4, np.pi/2, np.pi]:
        result = expr(
            params={"theta": theta_val},
            ctensor_data=[coeffs],
        )
        print(f"  theta={theta_val:.2f}: result={result:.4f}")


def example_circuit_cutting():
    """Full circuit cutting workflow."""
    print("\n" + "=" * 60)
    print("Example 3: Full Circuit Cutting Workflow")
    print("=" * 60)
    
    from qtpu import cut, circuit_to_hybrid_tn
    
    # Original circuit (too large for small QPU)
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.measure_all()
    
    print(f"Original circuit: {circuit.num_qubits} qubits")
    
    # Cut the circuit
    cut_circuit = cut(circuit, max_qubits=2)
    
    # Convert to HybridTensorNetwork
    htn = circuit_to_hybrid_tn(cut_circuit)
    
    print(f"\nAfter cutting:")
    print(f"  Subcircuits: {len(htn.qtensors)}")
    print(f"  QPD tensors: {len(htn.ctensors)}")
    
    for i, qt in enumerate(htn.qtensors):
        print(f"  Q{i}: shape {qt.shape}, inds {qt.inds}")
    
    for i, ct in enumerate(htn.ctensors):
        print(f"  C{i}: shape {ct.shape}, inds {ct.inds}")
    
    # Compile to callable
    expr = compile_htn(htn)
    
    print(f"\nCompiled: {expr.signature()}")
    
    # The ctensor_data comes from the QPD decomposition
    # In real usage, these are the coefficients already in htn.ctensors
    ctensor_data = [ct.data for ct in htn.ctensors]
    
    result = expr(ctensor_data=ctensor_data)
    
    print(f"\nResult: {result}")


def example_multiple_classical():
    """Multiple classical tensors with different roles."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple Classical Tensors")
    print("=" * 60)
    
    # Two indices: i for QPD, j for output selection
    idx_i = Parameter("i")
    idx_j = Parameter("j")
    
    def circuit_selector_i(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(k * np.pi / 3, 0)
        return qc
    
    def circuit_selector_j(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.rx(k * np.pi / 4, 0)
        return qc
    
    iswitch_i = ISwitch(idx_i, num_qubits=1, size=6, selector=circuit_selector_i)
    iswitch_j = ISwitch(idx_j, num_qubits=1, size=4, selector=circuit_selector_j)
    
    circuit = QuantumCircuit(1, 1)
    circuit.append(iswitch_i, [0])
    circuit.append(iswitch_j, [0])
    circuit.measure(0, 0)
    
    qt = QuantumTensor(circuit)
    print(f"Circuit tensor: shape {qt.shape}, inds {qt.inds}")
    
    # Two classical tensors:
    # 1. QPD coefficients (index i)
    # 2. Output weights (index j)
    qpd_spec = qtn.Tensor(np.zeros(6), inds=("i",))
    weights_spec = qtn.Tensor(np.zeros(4), inds=("j",))
    
    htn = HybridTensorNetwork(
        qtensors=[qt],
        ctensors=[qpd_spec, weights_spec],
    )
    
    evaluator = ExpvalEvaluator()
    expr = heinsum(htn, evaluator)
    
    print(f"\nExpression: {expr.signature()}")
    print(f"Classical tensor specs:")
    for i, spec in enumerate(expr.ctensor_specs):
        print(f"  {i}: inds={spec.inds}, shape={spec.shape}")
    
    # Execute with actual data
    qpd_coeffs = np.array([0.5, 0.5, 0.25, 0.25, -0.25, -0.25])
    output_weights = np.array([1.0, 0.5, 0.25, 0.125])
    
    result = expr(ctensor_data=[qpd_coeffs, output_weights])
    
    print(f"\nResult: {result}")


def example_output_indices():
    """Keep some indices in output (partial contraction)."""
    print("\n" + "=" * 60)
    print("Example 5: Partial Contraction (Output Indices)")
    print("=" * 60)
    
    # Circuit with two indices
    idx_i = Parameter("i")
    idx_j = Parameter("j")
    
    def selector_i(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(k * np.pi / 2, 0)
        return qc
    
    def selector_j(k: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.rx(k * np.pi / 3, 0)
        return qc
    
    iswitch_i = ISwitch(idx_i, num_qubits=1, size=3, selector=selector_i)
    iswitch_j = ISwitch(idx_j, num_qubits=1, size=4, selector=selector_j)
    
    circuit = QuantumCircuit(1, 1)
    circuit.append(iswitch_i, [0])
    circuit.append(iswitch_j, [0])
    circuit.measure(0, 0)
    
    qt = QuantumTensor(circuit)
    
    # Only contract over i, keep j as output
    coeffs_i = qtn.Tensor(np.zeros(3), inds=("i",))
    
    htn = HybridTensorNetwork(qtensors=[qt], ctensors=[coeffs_i])
    
    evaluator = ExpvalEvaluator()
    
    # Contract over i, output has index j
    expr = heinsum(htn, evaluator, output_inds="j")
    
    print(f"Expression: {expr.signature()}")
    
    coeffs = np.array([0.5, 0.3, 0.2])
    result = expr(ctensor_data=[coeffs])
    
    print(f"Result shape: {result.shape}")  # Should be (4,)
    print(f"Result: {result}")


def main():
    print("\n" + "=" * 60)
    print("HEinsum API")
    print("=" * 60)
    print("""
Design Philosophy:
- HybridTensorNetwork defines STRUCTURE (circuits, tensor indices, shapes)
- Classical tensors are TensorSpecs (no data, just structure)
- heinsum() compiles to callable HEinsum
- Call with params (circuit bindings) + ctensor_data (actual arrays)

This separates WHAT to compute from HOW to compute it:
- Structure is defined once (from circuit cutting, etc.)
- Data can vary at call time (different parameters, coefficients)
    """)
    
    example_basic()
    example_parameterized()
    example_circuit_cutting()
    example_multiple_classical()
    example_output_indices()


if __name__ == "__main__":
    main()
