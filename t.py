# from qiskit.circuit.library import EfficientSU2
# from qiskit.circuit import QuantumCircuit
# from qiskit.compiler import transpile
# from qiskit.transpiler import CouplingMap


# from qvm.cut_library.util import cut_qubit_connections
# from qvm.virt_router import instantiate


# cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])

# circuit = QuantumCircuit(2)
# circuit.cx(1, 0)
# c = transpile(circuit, optimization_level=3, coupling_map=cm, initial_layout=[0, 1])
# print(c)

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import FakeLagosV2

backend = FakeLagosV2()
pass_manager = generate_preset_pass_manager(2, backend)
print(pass_manager.passes())

# circ = QuantumCircuit(2)
# circ.cx(0, 1)
# circ.cx(1, 0)
# # circ = circ.decompose()
# circ = cut_qubit_connections(circ, {(circ.qubits[0], circ.qubits[1])})


# from qiskit.converters import circuit_to_dag
# print(circ)

# dags = []
# for circ in instantiate(circ):
#     c = transpile(circ, basis_gates=["rz", "h"], optimization_level=3)
#     print(c)
#     dag = circuit_to_dag(c)
#     if dag not in dags:
#         dags.append(dag)
        
# print(len(dags))


# circ = transpile(circ, basis_gates=["ecr", "rz", "sx"])
# print(circ)
