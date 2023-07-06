import qiskit
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


class QSVM:
    def __init__(self, qubit_count):
        self.qubit_count = qubit_count
        self.quantum_register = qiskit.circuit.QuantumRegister(self.qubit_count)
        self.circuit = qiskit.circuit.QuantumCircuit(self.quantum_register)
        self.hadamard_circuit()
        self.phase_addition()
        self.hadamard_circuit()
        self.circuit.measure_all()
        # self.circuit.draw(output="mpl")
        # try:
        #     plt.tight_layout()
        #     if not os.path.isdir("circuit_svg"):
        #         os.mkdir("circuit_svg")
        #     plt.tight_layout()
        #     plt.savefig(f"circuit_svg/qsvn_n{self.circuit.num_qubits}.png")
        # except:
        #     print("Figure too large - Can't save")

    def hadamard_circuit(self):
        for qubit in self.quantum_register:
            self.circuit.h(qubit)

    def phase_addition(self):
        for qubit in self.quantum_register:
            self.circuit.p(np.random.rand() * np.pi, qubit)
        for cqubit, aqubit in zip(
            self.quantum_register[:-1], self.quantum_register[1:]
        ):
            self.circuit.rzz(np.random.rand() * np.pi, cqubit, aqubit)
        iterables = list(self.quantum_register).copy()
        iterables.reverse()
        l1 = iterables[:-1]
        l2 = iterables[1:]
        for a1, a2 in zip(l1, l2):
            self.circuit.rzz(np.random.rand() * np.pi, a2, a1)
        for qubit in self.quantum_register:
            self.circuit.rz(np.random.rand() * np.pi, qubit)


name = "qsvm"

dirname = f"bench/circuits/{name}"
os.makedirs(dirname, exist_ok=True)

circuits = [QSVM(i).circuit for i in range(4, 30, 2)]
circuits += [QSVM(i).circuit for i in range(30, 101, 10)]

for circ in circuits:
    with open(f"{dirname}/{circ.num_qubits}.qasm", "w") as f:
        f.write(circ.qasm())
