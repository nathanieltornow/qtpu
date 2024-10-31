import numpy as np

# Define the given vectors again after the code execution reset
v1 = np.array([0, 3, 0], dtype=float)
v2 = np.array([1, -3, 0], dtype=float)
v3 = np.array([2, 4, -2], dtype=float)

v1 = np.array([-2, 0, 2], dtype=float)
v2 = np.array([2, 1, 0], dtype=float)
v3 = np.array([3, 6, 5], dtype=float)

# Function to perform Gram-Schmidt orthogonalization
def gram_schmidt(vectors):
    # Initialize an empty list for orthonormal vectors
    orthonormal_vectors = []
    
    for v in vectors:
        # Start with the current vector
        u = v.copy()
        # Subtract projections onto previously computed orthonormal vectors
        for u_i in orthonormal_vectors:
            projection = (np.dot(u_i, v) / np.dot(u_i, u_i)) * u_i
            u -= projection
        # Normalize the orthogonal vector
        if np.linalg.norm(u) > 0:  # Check to avoid division by zero
            orthonormal_vectors.append(u / np.linalg.norm(u))
    
    return np.array(orthonormal_vectors)

# Perform Gram-Schmidt on the given vectors
vectors = [v1, v2, v3]
orthonormal_vectors = gram_schmidt(vectors)

print(orthonormal_vectors)
