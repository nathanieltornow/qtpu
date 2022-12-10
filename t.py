from qiskit import QuantumCircuit, transpile
from qiskit.providers.ibmq import IBMQ

# IBMQ.load_account()
# provider = IBMQ.get_provider(
#     hub="ibm-q-research-2", group="tu-munich-1", project="main"
# )

# backend = provider.get_backend("ibm_perth")

circuit = QuantumCircuit(2)

circuit.h(0)
circuit.x(1)
circuit.rzz(0.3, 0, 1)
circuit.cx(0, 1)
circuit.rz(0.3, 1)
circuit.cx(0, 1)
circuit.h(0)
circuit.x(1)
circuit.measure_all()

circuit.barrier(0, label="barrier12")

print(circuit)

t_circ = transpile(
    circuit, optimization_level=3, basis_gates=[ "i", "rz", "rzz", "sx", "x"]
)

print(t_circ)
