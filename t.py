from qiskit.transpiler import CouplingMap

cm = CouplingMap.from_heavy_hex(21)
print(cm.size())