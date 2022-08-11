from qvm.virtual_circuit.fragment import Fragment
from qvm.circuit import Circuit

circ = Circuit(num_qubits=2, num_clbits=2)
f1 = Fragment(circ, {0, 1})

print(hash(f1))
