from qvm.compiler.cut.metis import *
from qvm.compiler.cut.asp import ASPCutter
from qvm.compiler.cut.cutter import *

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal

circuit = TwoLocal(
    8,
    rotation_blocks=["rz", "ry"],
    entanglement_blocks="rzz",
    entanglement="linear",
    reps=1,
).decompose()
c2 = MetisTNCutter(2).run(circuit)
# c2 = MetisQubitGraphCutter(2).run(circuit)
print(c2)
# c = wire_cuts_to_moves(c)
# print(decompose_circuit(c))
