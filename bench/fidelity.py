from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

from qvm.quasi_distr import QuasiDistr


def perfect_distr(circuit: QuantumCircuit):
    sim = AerSimulator(method="statevector")
    counts = sim.run(circuit, shots=20000).result().get_counts()
    return QuasiDistr.from_counts(counts, shots=20000)


def calcultate_fidelity(circuit: QuantumCircuit, noisy_distr: QuasiDistr) -> float:
    return hellinger_fidelity(perfect_distr(circuit), noisy_distr)


def total_variation_distance(p: QuasiDistr, q: QuasiDistr):
    events = set(p.keys()).union(set(q.keys()))
    tv_distance = 0.0
    for event in events:
        p_prob = p.get(event, 0.0)
        q_prob = q.get(event, 0.0)
        tv_distance += 0.5 * abs(p_prob - q_prob)
    return tv_distance


def calculate_total_variation_distance(
    circuit: QuantumCircuit, noisy_distr: QuasiDistr
) -> float:
    return total_variation_distance(perfect_distr(circuit), noisy_distr)
