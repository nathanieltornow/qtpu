from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from vqc.circuit import DistributedCircuit
from vqc.virtual_gate import VirtualCX
from vqc.executor import execute

circuit = QuantumCircuit(2)
circuit.h(0)
# append a virtual gate to the circuit
# virtual gates are subclasses of barriers
circuit.append(VirtualCX(), [0, 1])
circuit.measure_all()
print(circuit)

# create a distributed circuit of the circuit with
# virtual gates. The distributed circuit has multiple fragments,
# each fragment is represented as a quantum register.
dist_circ = DistributedCircuit.from_circuit(circuit)
print(dist_circ)
print(dist_circ.fragments)

# execute the distributed circuit (both fragments are executed independently)
counts = execute(dist_circ, default_backend=AerSimulator(), shots=1000)
print(counts)
