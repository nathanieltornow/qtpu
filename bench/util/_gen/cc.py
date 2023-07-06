import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# from qiskit import execute, Aer
import sys
import math
import os
import random


random.seed(555)


def gen_cc(qc, qr, cr, nCoins):
    indexOfFalseCoin = random.randint(0, nCoins - 1)

    for i in range(nCoins):
        qc.h(qr[i])
    for i in range(nCoins):
        qc.cx(qr[i], qr[nCoins])
    qc.measure(qr[nCoins], cr[nCoins])

    qc.x(qr[nCoins]).c_if(cr, 0)
    qc.h(qr[nCoins]).c_if(cr, 0)

    for i in range(nCoins):
        qc.h(qr[i]).c_if(cr, 2**nCoins)
    qc.barrier()

    qc.cx(qr[indexOfFalseCoin], qr[nCoins]).c_if(cr, 0)
    qc.barrier()

    for i in range(nCoins):
        qc.h(qr[i]).c_if(cr, 0)

    for i in range(nCoins):
        qc.measure(qr[i], cr[i])
    return qc


def cc(num_qubits):
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qr, cr)

    return gen_cc(qc, qr, cr, num_qubits - 1)


name = "cc"

dirname = f"bench/circuits/{name}"
os.makedirs(dirname, exist_ok=True)

circuits = [cc(i) for i in range(4, 30, 2)]
circuits += [cc(i) for i in range(30, 101, 10)]

for circ in circuits:
    with open(f"{dirname}/{circ.num_qubits}.qasm", "w") as f:
        f.write(circ.qasm())
