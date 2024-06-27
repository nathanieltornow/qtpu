from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator
from mpi4py import MPI

def Hpauli(dict,N):
    tableop = ""
    tableind = []
    for index, Pauli in sorted(dict.items(),reverse=True):
        tableop+=Pauli
        tableind.append(index)
    operator = SparsePauliOp.from_sparse_list([(tableop, tableind, 1)], num_qubits = N)
    return operator.simplify()




def create_ghz_circuit(n_qubits):
    ghz = QuantumCircuit(n_qubits)
    ghz.h(0)
    for qubit in range(n_qubits - 1):
        ghz.cx(qubit, qubit + 1)
    ghz.h(0)
    for qubit in range(n_qubits - 1):
        ghz.cx(qubit, qubit + 1)
    
    ghz.h(0)
    for qubit in range(n_qubits - 1):
        ghz.cx(qubit, qubit + 1)
    ghz.h(0)
    for qubit in range(n_qubits - 1):
        ghz.cx(qubit, qubit + 1)
    return ghz

size=28
circuit = create_ghz_circuit(size)

dicts = {}
for i in range(size):
    dicts[i]='Z'

result = estimator.run(circuit, "Z"*size).result()
if MPI.COMM_WORLD.Get_rank() == 0:
    print(result.values[0])