from qiskit.circuit import QuantumCircuit, Qubit, QuantumRegister
from qiskit.qasm3 import dumps


qreg = QuantumRegister

circuit = QuantumCircuit()

print(dumps(circuit))
