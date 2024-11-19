from typing import Callable

import numpy as np
from qiskit.circuit import QuantumCircuit

from qtpu.transforms import wire_cuts_to_moves, circuit_to_hybrid_tn


def esp(circuit: QuantumCircuit) -> float:
    fid = 1.0
    for instr in circuit:
        op = instr.operation

        if op.name == "barrier":
            continue

        if op.name == "measure":
            fid *= 1 - 1e-3

        elif op.num_qubits == 1:
            fid *= 1 - 1e-4

        elif op.num_qubits == 2:
            fid *= 1 - 1e-3

        else:
            raise ValueError(f"Unsupported operation: {op}")

    return round(fid, 3)


def estimated_error(circuit: QuantumCircuit) -> float:
    circuits = circuit_to_hybrid_tn(wire_cuts_to_moves(circuit)).subcircuits
    return np.mean([1 - esp(c) for c in circuits])
