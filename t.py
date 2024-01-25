# from dataclasses import dataclass

# import numpy as np
# import cotengra as ctg
# import networkx as nx

# from qvm.cut.contraction_tree import ContractionTree
# from qiskit.circuit.library import EfficientSU2
# from qvm.cut.success_estimator import QPUSizeEstimator
# from qvm.cut.compiler import cut
# from qvm.cut.bisectors.girvan_newman import GirvanNewmanBisector


# cricuit = EfficientSU2(6, reps=3).decompose()

# circuit = cut(
#     cricuit,
#     QPUSizeEstimator(3),
#     bisector=GirvanNewmanBisector(),
#     max_contraction_cost=300,
#     alpha=0.5,
# )

# print(circuit)

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qvm.cut.girvan_newman import girvan_newman_cut_circuit
from qvm.cut.metis import metis_cut_circuit
from qvm.cut.contraction_tree import contraction_tree_cut
from qvm.cut.estimators.qpu_size import QPUSizeEstimator

circuit = EfficientSU2(6, reps=1).decompose()


# print(girvan_newman_cut_circuit(circuit=circuit, num_fragments=2))
# print(metis_cut_circuit(circuit=circuit, num_fragments=2))


print(contraction_tree_cut(circuit=circuit, success_estimator=QPUSizeEstimator(3)))

# tree = CircuitContractionTree(cricuit)

# print(tree.circuit)

# tree.bisect()
# tree.bisect()

# print(tree.circuit)
# print(tree.contraction_cost())
# print(tree.left.circuit)
# print(tree.right.circuit)
