from qiskit import QuantumCircuit, execute, BasicAer
from qiskit.quantum_info.analysis import hellinger_fidelity

qc = QuantumCircuit(5, 5)
qc.h(2)
qc.cx(2, 1)
qc.cx(2, 3)
qc.cx(3, 4)
qc.cx(1, 0)
qc.measure(range(5), range(5))

sim = BasicAer.get_backend('qasm_simulator')
res1 = execute(qc, sim, shots=1000).result()
res2 = execute(qc, sim, shots=1212).result()

print(hellinger_fidelity(res1.get_counts(), res2.get_counts()) == hellinger_fidelity(res2.get_counts(), res1.get_counts()))
print(hellinger_fidelity(res1.get_counts(), res2.get_counts()))