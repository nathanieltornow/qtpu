from qiskit import QuantumCircuit

from bench.circuits import get_circuits
from qvm.compiler.virtualization.reduce_deps import (
    GreedyDependencyBreaker,
    QubitDependencyMinimizer,
)

from qvm.compiler.dag import DAG

circuit = get_circuits("twolocal_2", (5, 11))[0]

cut_circuit = QubitDependencyMinimizer(2).run(circuit)

print(cut_circuit)

print(DAG(circuit).num_dependencies())
print(DAG(cut_circuit).num_dependencies())

# print(circuit)
# print(cut_circuit)

# from qvm.compiler.virtualization.reduce_deps import GreedyDependencyBreaker

# circuit = GreedyDependencyBreaker(1).run(circuit)
# print(circuit)
# circuit = GreedyDependencyBreaker(1).run(circuit)
# print(circuit)
# circuit = GreedyDependencyBreaker(1).run(circuit)
# print(circuit)
# circuit = GreedyDependencyBreaker(1).run(circuit)
# print(circuit)
# from qiskit import transpile, schedule
# from qiskit.providers.fake_provider import FakeOslo

# circuit = transpile(circuit, FakeOslo(), optimization_level=0)

# G = nx.complete_graph(20)

# import matplotlib.pyplot as plt

# nx.draw(G)
# plt.show()
