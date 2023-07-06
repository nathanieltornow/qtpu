import qiskit
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
from itertools import permutations
import os


class VQC:
    def __init__(self, qubit_count, layers):
        self.qubit_count = qubit_count
        self.quantum_register = qiskit.circuit.QuantumRegister(self.qubit_count)
        # self.classical_register = qiskit.circuit.ClassicalRegister(self.qubit_count)
        self.circuit = qiskit.circuit.QuantumCircuit(self.quantum_register)
        self.layer_count = layers
        self.hadamard_circuit()
        self.phase_addition()
        self.learnable_layers()
        self.circuit.measure_all()
        # self.circuit.draw(output="mpl")
        # try:
        #     if not os.path.isdir("circuit_svg"):
        #         os.mkdir("circuit_svg")
        #     plt.tight_layout()
        #     plt.savefig(f"circuit_svg/vqc_n{k}.svg")
        # except:
        #     print("Fig too large - cant save")

    def hadamard_circuit(self):
        for qubit in self.quantum_register:
            self.circuit.h(qubit)

    def phase_addition(self):
        for qubit in self.quantum_register:
            self.circuit.rz(np.random.rand() * np.pi, qubit)
        for cqubit, aqubit in zip(
            self.quantum_register[:-1], self.quantum_register[1:]
        ):
            self.circuit.rzz(np.random.rand() * np.pi, cqubit, aqubit)

    def learnable_layers(self):
        for _ in range(self.layer_count):
            for qubit in self.quantum_register:
                self.circuit.ry(np.random.rand() * np.pi, qubit)
                self.circuit.rz(np.random.rand() * np.pi, qubit)
            qbs = list(self.quantum_register)
            for i, qb in enumerate(qbs):
                for j in range(i + 1, self.qubit_count):
                    self.circuit.cz(qb, qbs[j])


name = "vqc"

for layer in [1, 2, 3]:
    dirname = f"bench/circuits/{name}_{layer}"
    os.makedirs(dirname, exist_ok=True)

    circuits = [VQC(i, layer).circuit for i in range(6, 30, 2)]
    circuits += [VQC(i, layer).circuit for i in range(30, 101, 10)]

    for circ in circuits:
        with open(f"{dirname}/{circ.num_qubits}.qasm", "w") as f:
            f.write(circ.qasm())
