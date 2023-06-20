from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

# Save your credentials on disk.
# QiskitRuntimeService.save_account(
#     channel="ibm_quantum",
#     token="",
#     overwrite=True,
# )

service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q-research-2/tu-munich-1/main",
)

circuit = QuantumCircuit(2, 3)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure(0, 0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

from qiskit.providers.ibmq import IBMQ
from qiskit_aer.noise import NoiseModel

backend = IBMQ.load_account().get_backend("ibm_perth")
noise_model = NoiseModel.from_backend(backend)

from qiskit.compiler import transpile

session = Session(service=service, backend="ibmq_qasm_simulator")
sampler = Sampler(session=session)
circuit = transpile(circuits=circuit, backend=backend, optimization_level=3)
job1 = sampler.run(circuit, shots=1000, noise_model=noise_model, resilience_level=0)

print(job1.result())
