import logging
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import hellinger_fidelity

from qvm import VirtualCircuit, execute
from qvm.transpiler.fragmented_circuit import FragmentedCircuit

logging.basicConfig(level=logging.CRITICAL)


# create a simple circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.s(1)
circuit.cx(0, 1)
circuit.x(0)
circuit.h(1)
# circuit.cx(0, 2)
# Don't add barriers
circuit.measure(0, 0)
circuit.measure(1, 1)
# circuit.measure(2, 2)

# convert the circuit to a virtual circuit
virt_circ = VirtualCircuit.from_circuit(circuit)
# virtualize the connection between qubit 0 and qubit 1
virt_circ.virtualize_connection(circuit.qubits[0], circuit.qubits[1])
# print the virtual circuit
print(virt_circ)
# print the fragements of the virtual circuit
print(virt_circ.fragments)

print(FragmentedCircuit(virt_circ))

# execute the virtual circuit
counts = execute(virtual_circuit=virt_circ, shots=10000)
print(f"virtual: {counts}")

# execute the circuit without virtualization for comparison
t_circ = transpile(circuit, backend=AerSimulator(), optimization_level=2)
real_counts = AerSimulator().run(t_circ, shots=10000).result().get_counts()
print(f"real: {real_counts}")


print(f"fidelity: {hellinger_fidelity(real_counts, counts)}")
