import quimb as qu
import quimb.tensor as qtn
import sparse
import numpy as np

# class SparseCOOTensor(qtn.Tensor):
#     def __init__(self, data, inds, tags=None, dtype=None):
#         # Convert dense data to sparse COO format if necessary
#         if not isinstance(data, sparse.COO):
#             data = sparse.COO(data)

#         # Initialize the parent Tensor with COO data
#         super().__init__(data, inds, tags=tags, dtype=dtype)

    # def transpose(self, *axes):
    #     """Transpose the COO data and update the indices."""
    #     transposed_data = self.data.transpose(axes)
    #     new_inds = [self.inds[i] for i in axes]
    #     return SparseCOOTensor(transposed_data, new_inds, self.tags, dtype=self.dtype)

    # def to_dense(self):
    #     """Convert the sparse tensor data to dense format."""
    #     return self.data.todense()

    # def __matmul__(self, other):
    #     """Override matrix multiplication to support COO format."""
    #     if isinstance(other, SparseCOOTensor):
    #         return SparseCOOTensor(self.data @ other.data, self.inds, self.tags, dtype=self.dtype)
    #     elif isinstance(other, (np.ndarray, sparse.COO)):
    #         return SparseCOOTensor(self.data @ other, self.inds, self.tags, dtype=self.dtype)
    #     else:
    #         raise TypeError("Unsupported multiplication for SparseCOOTensor.")

    # def copy(self):
    #     """Return a copy of the tensor."""
    #     return SparseCOOTensor(self.data.copy(), self.inds, self.tags, dtype=self.dtype)

    # def to_dense_tensor(self):
    #     """Convert to a dense Quimb tensor."""
    #     dense_data = self.to_dense()
    #     return qtn.Tensor(dense_data, inds=self.inds, tags=self.tags, dtype=self.dtype)

# Example Usage
# Create a small dense array and convert it to COO format



dense_array = np.array([[1, 0, 0], [0, 1, 2], [0, 3, 0]])
sparse_tensor = qtn.Tensor(sparse.COO.from_numpy(dense_array), inds=("a", "b"))
t2 = qtn.Tensor(sparse.COO.from_numpy(dense_array), inds=("c", "b"))

print("Sparse COO data:", sparse_tensor.data)

tn = qtn.TensorNetwork([sparse_tensor, t2])
x = tn.contract(all, optimize='auto-hq')
print(x.data)
# print("Dense representation:", sparse_tensor.to_dense())
# print("Original Quimb Tensor:", sparse_tensor.to_dense_tensor())
