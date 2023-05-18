import numpy as np
from qiskit.circuit import QuantumCircuit


def hamsim(num_qubits: int, total_time: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    hbar = 0.658212  # eV*fs
    jz = (
        hbar * np.pi / 4
    )  # eV, coupling coeff; Jz<0 is antiferromagnetic, Jz>0 is ferromagnetic
    freq = 0.0048  # 1/fs, frequency of MoSe2 phonon

    w_ph = 2 * np.pi * freq
    e_ph = 3 * np.pi * hbar / (8 * np.cos(np.pi * freq))

    for step in range(total_time):
        # Simulate the Hamiltonian term-by-term
        t = step + 0.5

        # Single qubit terms
        psi = -2.0 * e_ph * np.cos(w_ph * t) * 1 / hbar
        for i in range(num_qubits):
            circuit.h(i)
            circuit.rz(psi, i)
            circuit.h(i)

        # Coupling terms
        psi2 = -2.0 * jz * 1 / hbar
        for i in range(num_qubits - 1):
            circuit.rzz(psi2, i, i + 1)

    circuit.measure_all()
    return circuit
