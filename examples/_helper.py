import numpy as np
from qiskit.circuit.library import TwoLocal


def simple_circuit(n: int):
    circuit = TwoLocal(n, ["u"], "rzz", entanglement="linear", reps=2).decompose()
    circuit = circuit.assign_parameters(
        {param: np.random.rand() * np.pi / 2 for param in circuit.parameters}
    )
    circuit.measure_all()
    return circuit
