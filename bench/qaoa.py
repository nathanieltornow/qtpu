import networkx as nx
import numpy as np
from qiskit import QuantumCircuit

G = nx.binomial_graph(15, 0.3)


# We will also bring the different circuit components that
# build the qaoa circuit under a single function
def create_qaoa_circ(num_qubits: int, p: float = 0.2):
    
    G = nx.binomial_graph(num_qubits, p)
    
    nqubits = len(G.nodes())
    qc = QuantumCircuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    gamma = np.random.uniform(0, 2 * np.pi, 1)[0]
    beta = np.random.uniform(0, np.pi, 1)[0]

    for pair in sorted(G.edges(), key=lambda x: (x[0], x[1])):
        qc.rzz(gamma, pair[0], pair[1])
    # mixer unitary
    for i in range(0, nqubits):
        qc.rx(beta, i)

    qc.measure_all()

    return qc


print(create_qaoa_circ(G))
