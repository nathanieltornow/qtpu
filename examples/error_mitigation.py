"""PEC error mitigation using qTPU's hybrid tensor network.

Demonstrates how Probabilistic Error Cancellation (PEC) maps naturally
onto an hTN: each noisy gate is replaced by an ISwitch over Pauli basis
operations, and the quasi-probability coefficients become a cTensor.

Key insight: for N mitigated gates, the naive approach enumerates 4^N
circuit variants. qTPU encodes all variants in a single qTensor and
contracts them efficiently via HEinsum.

Usage:
    uv run python examples/error_mitigation.py
"""

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import IGate, XGate, YGate, ZGate

from qtpu import CTensor, HEinsum, HEinsumRuntime, ISwitch, QuantumTensor

PAULIS = [IGate(), XGate(), YGate(), ZGate()]


def pec_coefficients(error_rate: float = 0.01) -> np.ndarray:
    """QPD coefficients for depolarizing noise at the given error rate."""
    gamma = error_rate / (1 - error_rate)
    return np.array([1 + 3 * gamma, -gamma, -gamma, -gamma])


def pec_iswitch(gate: QuantumCircuit, idx: str) -> ISwitch:
    """Create an ISwitch over PEC basis operations for a single-qubit gate."""
    param = Parameter(idx)

    def selector(basis_idx: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.append(PAULIS[basis_idx], [0])
        qc.compose(gate, inplace=True)
        qc.append(PAULIS[basis_idx], [0])
        return qc

    return ISwitch(param, 1, 4, selector)


def main():
    # -- 1. Base circuit to mitigate ----------------------------------------
    base = QuantumCircuit(4, 4)
    base.h(0)
    base.cx(0, 1)
    base.ry(0.5, 2)
    base.cx(2, 3)
    base.measure(range(4), range(4))

    # -- 2. Replace first N single-qubit gates with PEC ISwitches -----------
    num_pec_gates = 3
    pec_circuit = QuantumCircuit(4, 4)
    gate_count = 0

    for instr in base:
        if gate_count < num_pec_gates and instr.operation.num_qubits == 1:
            gate_qc = QuantumCircuit(1)
            gate_qc.append(instr.operation, [0])
            iswitch = pec_iswitch(gate_qc, f"pec_{gate_count}")
            pec_circuit.append(iswitch, instr.qubits)
            gate_count += 1
        else:
            pec_circuit.append(instr.operation, instr.qubits, instr.clbits)

    # -- 3. Build the hTN ---------------------------------------------------
    #    qTensor shape: (4, 4, 4) -- one axis per mitigated gate
    #    cTensors: quasi-probability coefficients for each gate
    qtensor = QuantumTensor(pec_circuit)
    coeffs = pec_coefficients(error_rate=0.01)
    ctensors = [CTensor(coeffs, (f"pec_{i}",)) for i in range(num_pec_gates)]

    heinsum = HEinsum(
        qtensors=[qtensor], ctensors=ctensors,
        input_tensors=[], output_inds=(),
    )

    naive_variants = np.prod(qtensor.shape)
    print(f"qTensor shape       : {qtensor.shape}")
    print(f"Naive circuit count : {naive_variants} (4^{num_pec_gates})")
    print(f"Einsum expression   : {heinsum.einsum_expr}")

    # -- 4. Execute ---------------------------------------------------------
    runtime = HEinsumRuntime(heinsum, backend="cudaq")
    runtime.prepare()
    result, timing = runtime.execute()

    print(f"\nMitigated result    : {result}")
    print(f"Circuits executed   : {timing.num_circuits}")
    print(f"Total time          : {timing.total_time:.3f}s")


if __name__ == "__main__":
    main()
