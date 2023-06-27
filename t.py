

# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile


# from qiskit.providers.fake_provider import FakeOslo

# circuit = QuantumCircuit(3)
# circuit.h(0)
# for i in range(100000):
#     circuit.cx(0, 1)
# circuit.measure_all()


# t1 = transpile(circuit,  basis_gates=["cz", "h"], backend=FakeOslo(), optimization_level=3)

# from qvm._circuit_hash import circuit_hash
# from time import perf_counter

# t0 = perf_counter()
# print(circuit_hash(circuit))
# print(perf_counter() - t0)

import networkx as nx
import matplotlib.pyplot as plt

# Generate a power-law graph with density 0.5
num_nodes = 16  # Number of nodes in the graph
avg_degree = int(num_nodes * 0.1)  # Average degree for each node

# Create the power-law graph
graph = nx.barabasi_albert_graph(num_nodes, 4)

# Plot the graph
pos = nx.spring_layout(graph)  # Position nodes using a spring layout
nx.draw(graph, pos, with_labels=True, node_size=100)
plt.title("Power-Law Graph with Density 0.5")
plt.show()