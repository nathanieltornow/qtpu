from qvm.compiler.cut.metis import (
    MetisTNCutter,
    MetisQubitGraphCutter,
    MetisPortGraphCutter,
)
from qvm.compiler.cut.asp import ASPCutter

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2

# circuit = EfficientSU2(8, reps=4).decompose()
circuit = QuantumCircuit(4)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(2, 3)
circuit.cx(1, 2)
circuit.cx(0, 1)
circuit.cx(2, 3)

c = MetisTNCutter(2).run(circuit)
c2 = MetisQubitGraphCutter(2).run(circuit)
print(c2)
