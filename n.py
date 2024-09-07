import numpy as np
import quimb.tensor as qtn


# tensor = qtn.Tensor(np.array([0.25]*36), inds=["i", "j"], tags=["wire"])


# tn = qtn.TensorNetwork([tensor])

class MyTensor(qtn.Tensor):
    def __init__(self):
        data = np.array([0.25]*36)
        super().__init__(data, ["i"], None)

    def __new__(cls):
        return super().__new__(cls)


tn = qtn.TensorNetwork([MyTensor(), MyTensor()])

print(tn.contract(all, optimize="auto-hq"))



# tn = tn.replace_with_svd(["wire"], ["i"], 0.0)
# print(tn)

# print(tensor)


A = np.diag(np.array([1, 0, 2, 3, 6]))

print(A)

U, S, V = np.linalg.svd(A)
print(U)
print(S)
print(V)