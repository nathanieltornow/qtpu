# ----------------------------------------------------------------------
# NWQBench: Northwest Quantum Proxy Application Suite
# ----------------------------------------------------------------------
# Ang Li, Samuel Stein, James Ang.
# Pacific Northwest National Laboratory(PNNL), U.S.
# BSD Lincese.
# Created 04/19/2021.
# ----------------------------------------------------------------------

import numpy as np
from qiskit import QuantumCircuit

# from qiskit import execute, Aer
import sys
import os


def majority(qc, a, b, c):
    qc.cx(c, b)
    qc.cx(c, a)
    qc.ccx(a, b, c)


def unmaj(qc, a, b, c):
    qc.ccx(a, b, c)
    qc.cx(c, a)
    qc.cx(a, b)


def adder4(qc, a0, a1, a2, a3, b0, b1, b2, b3, cin, cout):
    majority(qc, cin, b0, a0)
    majority(qc, a0, b1, a1)
    majority(qc, a1, b2, a2)
    majority(qc, a2, b3, a3)
    qc.cx(a3, cout)
    unmaj(qc, a2, b3, a3)
    unmaj(qc, a1, b2, a2)
    unmaj(qc, a0, b1, a1)
    unmaj(qc, cin, b0, a0)


def adder(n_bits):
    # n_bits = int(sys.argv[1])
    if n_bits % 4 != 0 or n_bits <= 0:
        print("Number of adder bits should be a multiply of 4.\n")
        exit(0)

    n_qubits = n_bits * 2 + 2 + int(n_bits / 4) - 1
    qc = QuantumCircuit(n_qubits, n_qubits)

    # ===================== Initialization ====================
    # we compute a=1110, b=0001, cin=1 => a=0000,b=0001,cout=1

    # a[0:n_bits-1]=1110
    for i in range(1, n_bits):
        qc.x(i)

    # b=[n_bits:2*n_bits-1]=0001
    qc.x(n_bits)

    # cin[2*n_bits] = 1
    qc.x(2 * n_bits)

    # ===================== Adder ====================

    for i in range(0, n_bits, 4):
        adder4(
            qc,
            i,
            i + 1,
            i + 2,
            i + 3,
            i + n_bits,
            i + n_bits + 1,
            i + n_bits + 2,
            i + n_bits + 3,
            n_bits * 2 + int(i / 4),
            n_bits * 2 + int(i / 4) + 1,
        )

    qc.measure_all()
    return qc


name = "adder"

dirname = f"bench/circuits/{name}"
os.makedirs(dirname, exist_ok=True)

circuits = [adder(i) for i in range(4, 30, 4)]
# circuits += [adder(i) for i in range(30, 101, 10)]

for circ in circuits:
    with open(f"{dirname}/{circ.num_qubits}.qasm", "w") as f:
        f.write(circ.decompose().qasm())

# if not os.path.isdir("qasm"):
#     os.mkdir("qasm")
# qasm_file = open("qasm/" + "adder_n" + str(n_qubits) + ".qasm","w")
# qasm_file.write(qc.qasm())
# qasm_file.close()

# simulator = Aer.get_backend('qasm_simulator')
# job = execute(qc,simulator,shots=10)
# result = job.result()
# counts = result.get_counts(qc)
# print (counts)
