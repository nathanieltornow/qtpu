from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap, Layout
from qiskit.compiler import transpile
from qiskit.providers import BackendV1
from qiskit.providers.fake_provider import FakeOslo, FakeGuadalupe, FakeMontrealV2

from qvm.virt_router import run, route_circuit_trivial, virt_furthest_qubits, instantiate
from qvm.cut_library.decomposition import bisect

from circuits import qaoa, ghz, qft, twolocal, dj, random_circuit

def num_swaps(circuit: QuantumCircuit) -> int:
    return sum([1 for instr in circuit.data if instr.operation.name == "swap"])

def num_cnots(circuit: QuantumCircuit) -> int:
    return sum([1 for instr in circuit.data if instr.operation.name == "cx"])


# circuit = (7, 3)
# circuit = QuantumCircuit.from_qasm_str(qasm)
import networkx as nx

# circuit = qaoa(nx.complete_graph(7))
circuit = twolocal(14, 1)



import mapomatic as mm

# from qiskit.providers.ibmq import IBMQ

# provider = IBMQ.load_account()

backend = FakeGuadalupe()

t_circuit = transpile(
    circuit,
    backend=backend,
    # initial_layout=init_layout,
    optimization_level=3,
)


layout = t_circuit._layout.initial_layout.get_virtual_bits()

layout = [(q,p) for q, p in layout.items() if q in circuit.qubits]

def index_of(qubit):
    return circuit.find_bit(qubit).index

init_layout = [p for _, p in sorted(layout, key=lambda x: index_of(x[0]))]
print(init_layout)

# init_layout = [0, 1, 2, 4, 5, 6]

cm = CouplingMap(backend.configuration().coupling_map) if isinstance(backend, BackendV1) else backend.coupling_map

v_circuit = virt_furthest_qubits(circuit, init_layout, cm, 3)
# v_circuit = bisect(circuit, 3)
# v_circuit = circuit.copy()
# print(t_circuit)
# print(v_circuit)
inst0 = instantiate(v_circuit)[0]
inst0 = transpile(
    inst0,
    backend=backend,
    initial_layout=init_layout,
    optimization_level=3,
)
print(v_circuit)
print(num_cnots(t_circuit))
print(num_cnots(inst0))
# print(t_circuit)
# print(inst0)
# print(sum([1 for instr in v_circuit.data if instr.operation.]))
# res = run(circuit, 1000)
