import quimb.tensor as qtn
import numpy as np



O1 = qtn.Tensor(np.random.rand(6, 6), inds=('1', '2'))
O2 = qtn.Tensor(np.random.rand(6, 6, 4), inds=('1', '2', '3'))
O3 = qtn.Tensor(np.random.rand(6, 6, 4), inds=('5', '6', '3'))
O4 = qtn.Tensor(np.random.rand(6, 6), inds=('6', '5'))

tn = qtn.TensorNetwork([O1, O2, O3, O4])
# tn.draw(color=['O1', 'O2', 'O3', 'O4'])
print(tn.contraction_cost(optimize='greedy'))




# # createn six 6x6 random tensors, and put them in a network with a ring topology
# tensor = qtn.Tensor(np.random.rand(6, 8, 2), inds=('k0', 'k1', 'b0'))

# vec = qtn.Tensor(np.random.rand(6), inds=('k0',))

# tn = qtn.TensorNetwork([tensor, vec])

# print(tn.contraction_cost(optimize='auto-hq'))