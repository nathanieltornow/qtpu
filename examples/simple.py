from qiskit import QuantumCircuit
from qvm.circuit import VirtualCircuit

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.s(1)
circuit.cx(0, 1)
circuit.x(0)
circuit.h(1)
circuit.measure_all()

