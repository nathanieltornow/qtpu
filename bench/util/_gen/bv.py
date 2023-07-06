import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# from qiskit import execute, Aer
import sys
import math
import random
import os


def generate_astring(nqubits, prob=1.0):
    answer = []
    for i in range(nqubits):
        if random.random() <= prob:
            answer.append("1")
        else:
            answer.append("0")
    return "".join(answer)


def gen_bv(qc, qr, cr, n_qubits):
    hiddenString = generate_astring(n_qubits - 1, 0.5)
    for i in range(n_qubits - 1):
        qc.h(qr[i])
    qc.x(qr[n_qubits - 1])
    qc.h(qr[n_qubits - 1])
    qc.barrier()
    hiddenString = hiddenString[::-1]
    for i in range(len(hiddenString)):
        if hiddenString[i] == "1":
            qc.cx(qr[i], qr[n_qubits - 1])
    hiddenString = hiddenString[::-1]
    qc.barrier()
    for i in range(n_qubits - 1):
        qc.h(qr[i])
    for i in range(n_qubits - 1):
        qc.measure(qr[i], cr[i])
    return qc


def bv(n_qubits: int) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)

    return gen_bv(qc, qr, cr, n_qubits)


name = "bv"

dirname = f"bench/circuits/{name}"
os.makedirs(dirname, exist_ok=True)

circuits = [bv(i) for i in range(4, 30, 2)]
circuits += [bv(i) for i in range(30, 101, 10)]

for circ in circuits:
    with open(f"{dirname}/{circ.num_qubits}.qasm", "w") as f:
        f.write(circ.qasm())


# if not os.path.isdir("qasm"):
#     os.mkdir("qasm")
# qasm_file = open("qasm/bv_n" + str(n_qubits) + ".qasm", "w")
# qasm_file.write(qc.qasm())
# qasm_file.close()
