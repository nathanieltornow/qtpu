from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

from qvm.quasi_distr import QuasiDistr


def calcultate_fidelity(circuit: QuantumCircuit, noisy_distr: QuasiDistr) -> float:
    
    sim = AerSimulator(method="statevector")
    counts = sim.run(circuit, shots=20000).result().get_counts()
    ideal_distr = QuasiDistr.from_counts(counts, shots=20000)
    return hellinger_fidelity(ideal_distr, noisy_distr)
