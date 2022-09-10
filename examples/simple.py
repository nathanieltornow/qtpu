from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qvm.circuit import DistributedCircuit
from qvm.virtual_gate import VirtualCX
from qvm.executor import execute

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.append(VirtualCX(), [0, 1])
circuit.measure_all()
print(circuit)


virt_circ = DistributedCircuit.from_circuit(circuit)
print(virt_circ)

for frag in virt_circ.fragments:
    print(virt_circ.fragment_as_circuit(frag))

counts = execute(virt_circ, default_backend=AerSimulator())
print(counts)
