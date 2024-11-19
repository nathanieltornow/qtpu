"""Helper functions for the QTPU module."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def nearest_probability_distribution(quasi_dist: Iterable[float]) -> dict[int, float]:
    """Adjusts a quasi-probability distribution to the nearest valid probability distribution.

    Args:
        quasi_dist (Iterable[float]): An iterable of quasi-probabilities.

    Returns:
        dict[int, float]: A dictionary where keys are indices of the original quasi-probabilities
                          and values are the adjusted probabilities forming a valid distribution.
    """
    probs = dict(enumerate(quasi_dist))

    sorted_probs = dict(sorted(probs.items(), key=operator.itemgetter(1)))

    num_elems = len(sorted_probs)
    new_probs = {}
    beta = 0.0
    diff = 0.0

    for key, val in sorted_probs.items():
        temp = val + beta / num_elems
        if temp < 0:
            beta += val
            num_elems -= 1
            diff += val * val
        else:
            diff += (beta / num_elems) * (beta / num_elems)
            new_probs[key] = val + beta / num_elems

    return new_probs


# def qiskit_to_quimb(circuit: QuantumCircuit) -> qtn.Circuit:
#     circ = qtn.Circuit(circuit.num_qubits)
#     for instr in circuit:
#         op, qubits = instr.operation, instr.qubits
#         if not isinstance(op, Gate):
#             continue

#         circ.apply_gate_raw(op.to_matrix(), [circuit.qubits.index(q) for q in qubits])

#     return circ


# def sample_quimb(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
#     circuit.remove_final_measurements()

#     tn_circ = qiskit_to_quimb(circuit)

#     counts = {}
#     for sample in tn_circ.sample(shots):
#         sample_str = "".join(reversed(sample))
#         counts[sample_str] = counts.get(sample_str, 0) + 1
#     return counts


# def expval_quimb(circuit: QuantumCircuit) -> float:
#     tn_circ = qiskit_to_quimb(circuit)
#     Z = qu.pauli("Z")
#     for i in range(circuit.num_qubits - 1):
#         Z = Z & qu.pauli("Z")
#     return tn_circ.local_expectation(Z, range(circuit.num_qubits))


# def compute_Z_expectation(circuit: QuantumCircuit) -> float:
#     from cuquantum import CircuitToEinsum, contract
#     import cupy as cp

#     myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
#     pauli_string = "Z" * circuit.num_qubits
#     expression, operands = myconverter.expectation(pauli_string, lightcone=True)
#     return contract(expression, *operands)
