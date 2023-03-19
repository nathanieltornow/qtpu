from qiskit import QuantumCircuit, transpile
from math import pi, cos, sin

circuit = QuantumCircuit(2)
circuit.rzz(pi, 0, 1)
circuit.h(0)
circuit.h(0)
circuit = transpile(circuit, optimization_level=3)
print(circuit)

# provider = IBMQ.load_account()
# backend = provider.get_backend('ibmq_montreal')
# noise_model = NoiseModel.from_backend(backend)
# print("noise model: ", noise_model)

# coupling_map = backend.configuration().coupling_map
# basis_gates = noise_model.basis_gates


# circ = QuantumCircuit(5)
# circ.h(0)
# circ.cx([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])
# circ.measure_all()

# print("running on sim")
# simulator = provider.get_backend('ibmq_qasm_simulator')
# results = simulator.run(circ, shots=1000, noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates).result()
# print(results.get_counts())
