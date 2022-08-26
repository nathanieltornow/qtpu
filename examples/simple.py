from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import hellinger_fidelity

from qvm import VirtualCircuit, execute_virtual_circuit


# create a simple circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.s(1)
circuit.cx(0, 1)
circuit.x(0)
circuit.h(1)
# Don't add barriers
circuit.measure(0, 0)
circuit.measure(1, 1)

# convert the circuit to a virtual circuit
virt_circ = VirtualCircuit.from_circuit(circuit)
# virtualize the connection between qubit 0 and qubit 1
virt_circ.virtualize_connection(circuit.qubits[0], circuit.qubits[1])
# print the virtual circuit
print(virt_circ)

# execute the virtual circuit
# specify the backend and how you want the configurations
# to be transpiled and executed
counts = execute_virtual_circuit(
    virtual_circuit=virt_circ,
    backend=AerSimulator(),
    transpile_flags={"optimization_level": 2},
    exec_flags={"shots": 1024},
)
print(f"virtual: {counts}")

# execute the circuit without virtualization for comparison
t_circ = transpile(circuit, backend=AerSimulator(), optimization_level=2)
real_counts = AerSimulator().run(t_circ, shots=1024).result().get_counts()
print(f"real: {real_counts}")


print(f"fidelity: {hellinger_fidelity(real_counts, counts)}")
