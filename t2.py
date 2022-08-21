from time import sleep
from qiskit import IBMQ, transpile
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator

# Build a thousand circuits.
circs = []
for _ in range(2):
    circs.append(random_circuit(num_qubits=5, depth=4, measure=True))

backend = AerSimulator()
# Need to transpile the circuits first.
circs = transpile(circs, backend=backend)

# Use Job Manager to break the circuits into multiple jobs.
job_manager = IBMQJobManager()
job_set_foo = job_manager.run(circs, backend=backend, name="foo")

print(job_set_foo.status())

sleep(113)