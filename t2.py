from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import HGate

from qvm.circuit import VirtualCircuit


circuit = QuantumCircuit(3)
circuit.h(0)
circuit.measure_all()
print(VirtualCircuit(circuit))
