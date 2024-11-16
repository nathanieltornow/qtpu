import numpy as np

def V(phi):
    return np.sqrt(1 - 3 * np.cos(phi)**2 + 2 * np.cos(phi)**3)


print(V(np.pi/3))