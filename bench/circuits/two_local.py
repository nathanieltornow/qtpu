import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, EfficientSU2


def two_local(
    num_qubits: int, reps: int = 1, entanglement: str = "circular"
) -> QuantumCircuit:
    """Returns a quantum circuit implementing EfficientSU2 ansatz with random parameter
    values.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    """

    np.random.seed(10)
    qc = TwoLocal(
        num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="rzz",
        entanglement=entanglement,
        reps=reps,
    )
    num_params = qc.num_parameters
    qc = qc.bind_parameters(np.random.rand(num_params))
    qc.measure_all()
    qc.name = "twolacalrandom"

    return qc.decompose()
