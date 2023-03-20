import numpy as np
from qiskit.circuit.library import EfficientSU2


def test_bisection():
    num_qubits = 4
    circuit = EfficientSU2(
        num_qubits=num_qubits,
        reps=1,
        entanglement="linear",
        su2_gates=["ry"],
    )
    circuit.measure_all()
    circuit = circuit.decompose()

    params = [(np.pi * i) / 16 for i in range(len(circuit.parameters))]
    circuit = circuit.bind_parameters(params)
