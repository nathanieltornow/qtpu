import logging

import numpy as np
from qiskit.circuit.library import TwoLocal

NUM_QUBITS = 6
NUM_REPS = 3


def example_circuit():
    # create your quantum circuit with Qiskit
    circuit = TwoLocal(
        NUM_QUBITS, ["h", "rz"], "cx", entanglement="linear", reps=NUM_REPS
    )
    circuit.measure_all()
    circuit = circuit.decompose()
    params = [(np.random.uniform(0.0, np.pi)) for _ in range(len(circuit.parameters))]
    return circuit.bind_parameters(params)
