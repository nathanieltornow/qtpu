from qvm.virtual_gate import VirtualCX
from qiskit import QuantumCircuit

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.append(VirtualCX(), (0, 1), ())
circuit.measure(0, 0)
circuit.measure(1, 1)
print(circuit)