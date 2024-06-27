import numpy as np

Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)





def isunitary(U):
    return np.allclose(np.eye(U.shape[0]), U.conj().T @ U)


U2 = np.exp(1j * 2 * np.pi * Z / 4)

U = np.kron((I + Z)/2, U2)
print(isunitary(U))