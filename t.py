from qiskit import QuantumCircuit
from copy import deepcopy

circuit = QuantumCircuit(2)
circuit.cx(0, 1)

qreg2 = deepcopy(circuit.qregs[0])
print(hash(qreg2) == hash(circuit.qregs[0]))
print(qreg2 == circuit.qregs[0])

from qiskit_aer import UnitarySimulator

print(UnitarySimulator().run(circuit).result().get_unitary())