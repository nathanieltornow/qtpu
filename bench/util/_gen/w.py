import numpy as np
from qiskit import QuantumCircuit
from qiskit import execute, Aer
import os
import sys
import random
import math


def F_gate(qc, i, j, n, k):
    theta = math.acos(math.sqrt(1.0 / (n - k + 1)))
    qc.ry(-theta, j)
    qc.cz(i, j)
    qc.ry(theta, j)


def w_state(num_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits - 1):
        F_gate(circuit, num_qubits - 1 - i, num_qubits - 2 - i, num_qubits, i + 1)

    for i in range(0, num_qubits - 1):
        circuit.cx(num_qubits - 2 - i, num_qubits - 1 - i)

    circuit.measure_all()
    return circuit


name = "wstate"

dirname = f"bench/circuits/{name}"
os.makedirs(dirname, exist_ok=True)

circuits = [w_state(i) for i in range(4, 30, 2)]
circuits += [w_state(i) for i in range(30, 101, 10)]

for circ in circuits:
    with open(f"{dirname}/{circ.num_qubits}.qasm", "w") as f:
        f.write(circ.qasm())


# qc = QuantumCircuit(n_qubits, n_qubits)
# n = n_qubits

# qc.x(n - 1)

# for i in range(0, n - 1):
#     F_gate(qc, n - 1 - i, n - 2 - i, n, i + 1)

# for i in range(0, n - 1):
#     qc.cx(n - 2 - i, n - 1 - i)

# qc.measure_all()
# if not os.path.isdir("qasm"):
#     os.mkdir("qasm")
# qasm_file = open("qasm/w_state_n" + str(n_qubits) + ".qasm", "w")
# qasm_file.write(qc.qasm())
# qasm_file.close()
