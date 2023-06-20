from qvm.compiler import cut_wires, cut_gates_optimal
from qvm.dag import DAG

from _example_circuit import example_circuit

# Cutting a circuit into halfs


circuit = example_circuit(num_qubits=10, num_reps=3, entanglement="linear")

print("Before cutting:")
print(circuit.draw())

wire_dag = DAG(circuit)

cut_wires(wire_dag, 5)
print("After wire cutting:")




