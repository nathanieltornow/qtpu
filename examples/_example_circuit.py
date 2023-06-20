import logging

import numpy as np
from qiskit.circuit.library import TwoLocal


def example_circuit(num_qubits: int, num_reps: int, entanglement: str):
    circuit = TwoLocal(
        num_qubits, ["rx"], "rzz", entanglement=entanglement, reps=num_reps
    )
    circuit.measure_all()
    circuit = circuit.decompose()
    params = [(np.random.uniform(0.0, np.pi)) for _ in range(len(circuit.parameters))]
    return circuit.bind_parameters(params)
