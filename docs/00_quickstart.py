from qiskit import QuantumCircuit
import qtpu
from qtpu.compiler.terminators import reach_num_qubits

N = 20
circuit = QuantumCircuit(N)
circuit.h(0)
circuit.cx(range(0, N-1), range(1, N))
circuit.measure_all()

# cut the circuit into a hybrid tensor network with quantum-tensors
# half the size of the original cricuit
hybrid_tn = qtpu.cut(circuit, terminate_fn=reach_num_qubits(N//2), max_cost=5)

for qtens in hybrid_tn.quantum_tensors:
    print(qtens.circuit)

# contract the hybrid tensor network running on both quantum and classical devices
result = qtpu.contract(hybrid_tn, shots=100000)
print(result)