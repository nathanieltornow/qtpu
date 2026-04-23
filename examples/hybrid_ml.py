"""Hybrid ML: quantum layer + classical linear layer via HEinsum.

Demonstrates the qTPU programming model from Listing 2 of the paper:
a quantum kernel with iswitch indices composed with a classical tensor.
"""

import torch
from qiskit.circuit import QuantumCircuit

from qtpu.core import QuantumTensor, ISwitch, HEinsum, CTensor

# --- Build a quantum layer as a qTensor ---

NQUBITS = 4
BATCH_SIZE = 8
N_OBS = 2


def build_quantum_layer() -> QuantumTensor:
    """A qTensor with batch index (i) and observable index (k)."""
    qc = QuantumCircuit(NQUBITS)

    # Input encoding: iswitch over batch dimension
    for q in range(NQUBITS):
        qc.append(
            ISwitch("i", [f"rx({x})" for x in torch.randn(BATCH_SIZE)], NQUBITS),
            [q],
        )

    # Trainable layer (shared across batch)
    for q in range(NQUBITS - 1):
        qc.cx(q, q + 1)

    # Measurement observable: iswitch over observable dimension
    qc.append(
        ISwitch("k", ["z"] * N_OBS, NQUBITS),
        list(range(NQUBITS)),
    )

    return QuantumTensor.from_circuit(qc)


# --- Compose into a hybrid computation ---
qt = build_quantum_layer()
V = CTensor(torch.randn(N_OBS, 3))  # classical weights: observables -> features

# HEinsum: contract quantum output (ik) with classical weights (kj) -> (ij)
heinsum = HEinsum("ik,kj->ij", qt, V)

print(f"Quantum tensor shape: {qt.shape}")
print(f"Classical tensor shape: {V.shape}")
print(f"Output shape: batch={BATCH_SIZE} x features=3")
