from qiskit.circuit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_optimization import QuadraticProgram

from .utils import get_examplary_max_cut_qp


def vqe(num_qubits: int, reps: int = 1) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Variational Quantum Eigensolver Algorithm for a specific max-cut
     example.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    """

    qp = get_examplary_max_cut_qp(num_qubits)
    assert isinstance(qp, QuadraticProgram)

    ansatz = RealAmplitudes(num_qubits, reps=reps)
    vqe = VQE(ansatz=ansatz, optimizer=SLSQP(maxiter=25), estimator=Estimator())
    vqe_result = vqe.compute_minimum_eigenvalue(qp.to_ising()[0])
    qc = vqe.ansatz.bind_parameters(vqe_result.optimal_point)

    qc.measure_all()
    qc.name = "vqe"

    return qc
