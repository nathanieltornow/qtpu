from qiskit.circuit import QuantumCircuit
import networkx as nx


def circ(n, d):
    circuit = QuantumCircuit(n, n)
    circuit.h(range(n))
    for _ in range(d):
        circuit.cx(range(n - 1), range(1, n))
    circuit.measure(range(n), range(n))
    return circuit


circuit = circ(8, 5)
print(circuit)

from qvm.dag import DAG

dag = DAG(circuit)


from qvm.compiler.gate_virt import (
    minimize_qubit_dependencies,
    bisect_recursive,
    optimal_gate_cut,
)
from qvm.compiler.wire_cut import cut_wires
from qvm.compiler.qubit_reuse import apply_maximal_qubit_reuse

cut_wires(dag, 5)
# minimize_qubit_dependencies(dag, 5)

# minimize_qubit_dependencies(dag, 2)
from qvm.virtual_gates import VirtualBinaryGate


# print(dag.to_circuit())


# dag.remove_nodes_of_type(VirtualBinaryGate)

# print(dag.to_circuit())

# apply_maximal_qubit_reuse(dag)

print(dag.to_circuit())
