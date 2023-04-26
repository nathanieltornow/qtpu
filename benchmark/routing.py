from circuits import dj, ghz, qaoa, qft, random_circuit, twolocal
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers import BackendV1
from qiskit.providers.fake_provider import (FakeGuadalupe, FakeMontrealV2,
                                            FakeOslo)
from qiskit.transpiler import CouplingMap, Layout

from qvm.cut_library.decomposition import bisect
from benchmark.virt_router import (instantiate, route_circuit_trivial,
                             virt_furthest_qubits)


def num_swaps(circuit: QuantumCircuit) -> int:
    return sum([1 for instr in circuit.data if instr.operation.name == "swap"])

def num_cnots(circuit: QuantumCircuit) -> int:
    return sum([1 for instr in circuit.data if instr.operation.name == "cx"])


# circuit = (7, 3)
# circuit = QuantumCircuit.from_qasm_str(qasm)
import networkx as nx
from numpy import pi

n = 8
circuit = QuantumCircuit(n)
for i in range(n):
    circuit.rz(pi/2, i)
for i in range(2, n, 2):
    circuit.rzz(pi/2, i-1, i)
for i in range(1, n, 2):
    circuit.rzz(pi/2, i-1, i)
for i in range(2, n, 2):
    circuit.rzz(pi/2, i-1, i)
    
circuit.rzz(pi/2, 0, n-1)
    
# print(circuit)

# exit(0)

# circuit = qaoa(nx
# circuit = qaoa(nx.ladder_graph(3))
# circuit = twolocal(6, 1)


# from qiskit.providers.ibmq import IBMQ

# provider = IBMQ.load_account()

backend = FakeMontrealV2()

# init_layout = [0, 1, 2, 4, 5, 6]

t_circuit = transpile(
    circuit,
    # backend=backend,
    coupling_map=backend.coupling_map,
    initial_layout=[7, 10, 12, 13, 14, 16, 19, 20],
    optimization_level=3,
)

# print(circuit)
print(t_circuit)
print(num_swaps(t_circuit))
# layout = t_circuit._layout.initial_layout.get_virtual_bits()

# layout = [(q,p) for q, p in layout.items() if q in circuit.qubits]

# def index_of(qubit):
#     return circuit.find_bit(qubit).index

# init_layout = [p for _, p in sorted(layout, key=lambda x: index_of(x[0]))]
# print(init_layout)
exit()
# layout = t_circuit._layout.initial_layout.get_virtual_bits()

# layout = [(q,p) for q, p in layout.items() if q in circuit.qubits]

# def index_of(qubit):
#     return circuit.find_bit(qubit).index

# init_layout = [p for _, p in sorted(layout, key=lambda x: index_of(x[0]))]
# print(init_layout)

# init_layout = [0, 1, 2, 4, 5, 6]

cm = CouplingMap(backend.configuration().coupling_map) if isinstance(backend, BackendV1) else backend.coupling_map
cm = CouplingMap.from_line(n)

v_circuit = virt_furthest_qubits(circuit, list(range(circuit.num_qubits)), cm, 2)
# v_circuit = bisect(circuit, 3)
# v_circuit = circuit.copy()
# print(t_circuit)
# print(v_circuit)
inst0 = instantiate(v_circuit)[0]
inst0 = transpile(
    inst0.decompose(),
    # backend=backend,
    coupling_map=cm, 
    # initial_layout=init_layout,
    optimization_level=1,
)
print(v_circuit)
print(num_swaps(t_circuit))
print(num_swaps(inst0))
# print(t_circuit)
# print(inst0)
# print(sum([1 for instr in v_circuit.data if instr.operation.]))
# res = run(circuit, 1000)
