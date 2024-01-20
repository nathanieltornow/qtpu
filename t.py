# import numpy as np
# import quimb.tensor as qtn


# def simple_example():
#     A = qtn.rand_tensor((2, 2), inds=["i", "j"])
#     C1 = qtn.rand_tensor((2), inds=["i"])
#     C2 = qtn.rand_tensor((2), inds=["j"])

#     tn1 = A & C1 & C2

#     sol1 = tn1.contract(all, optimize="auto-hq")

#     C = qtn.Tensor(np.tensordot(C1.data, C2.data, axes=0), inds=["i", "j"])

#     tn2 = A & C
#     sol2 = tn2.contract(all, optimize="auto-hq")

#     assert np.allclose(sol1, sol2)


# def complex_example():
#     A = qtn.rand_tensor((2, 2), inds=["i", "j"])
#     B = qtn.rand_tensor((2, 2), inds=["k", "l"])
#     C1 = qtn.rand_tensor((2, 2), inds=["i", "k"])
#     C2 = qtn.rand_tensor((2, 2), inds=["j", "l"])

#     tn1 = qtn.TensorNetwork([A, B, C1, C2])

#     sol1 = tn1.contract(all, optimize="auto-hq")
#     print(sol1)
#     C = qtn.Tensor(np.tensordot(C1.data, C2.data, axes=0), inds=["i", "k", "j", "l"])

#     tn2 = A & C & B
#     sol2 = tn2.contract(all, optimize="auto-hq")
#     print(sol2)
#     assert np.allclose(sol1, sol2)


# # simple_example()
# complex_example()

from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

circ = QuantumCircuit(2, 2)
circ.h(0)

counts = AerSimulator().run(circ, shots=1000).result().get_counts()
print(counts)
