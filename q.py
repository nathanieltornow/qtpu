from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

circuit = QuantumCircuit(2, 1)
circuit.h(0)
circuit.x(1)
circuit.cz(0, 1)
circuit.h(0)
circuit.measure(0, 0)


props = AerSimulator().run(circuit).result().get_counts()
print(props)