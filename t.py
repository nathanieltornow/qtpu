from qiskit import QuantumCircuit

from bench.circuits import get_circuits
from qvm.compiler.virtualization.reduce_deps import (
    GreedyDependencyBreaker,
    QubitDependencyMinimizer,
)
from qvm.compiler.qubit_reuse import QubitReuseCompiler
from qvm.virtual_circuit import VirtualCircuit
from qvm.compiler.dag import DAG

from qiskit_aer import StatevectorSimulator

circuit = QuantumCircuit(3, 3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)

print(StatevectorSimulator().run(circuit, shots=1000).result().get_counts())


circuit2 = circuit.copy()
circuit2.measure(0, 0)


circuit3 = circuit.copy()
circuit3.measure(1, 1)
circuit3.measure(2, 2)
print(StatevectorSimulator().run(circuit2, shots=1000).result().get_counts())
print(StatevectorSimulator().run(circuit3, shots=1000).result().get_counts())

# vc = VirtualCircuit(cut_circuit)
# QubitReuseCompiler(1).run(vc)

# c = list(vc.fragment_circuits.values())[0]
# print(c)

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
