import logging

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator, StatevectorSimulator

from qiskit.providers.ibmq import IBMQ
from qiskit.compiler import transpile


from qvm.runtime.util import sample_on_ibmq_backend
from qvm.bench import fidelity

from qvm.stack._types import QVMJobMetadata
from qvm.stack.decomposer import Decomposer
from qvm.stack.qpu_runner import QPURunner


def get_circuit(num_qubits: int) -> QuantumCircuit:
    circuit = EfficientSU2(
        num_qubits=num_qubits,
        reps=3,
        entanglement="linear",
        su2_gates=["rx", "ry", "rz"],
    )
    circuit.measure_all()
    circuit = circuit.decompose()
    params = [(np.pi * i) / 16 for i in range(len(circuit.parameters))]
    circuit = circuit.bind_parameters(dict(zip(circuit.parameters, params)))
    return circuit


def main():
    circuit = get_circuit(8)
    # circuit = QuantumCircuit(2)
    # circuit.h(0)
    # circuit.h(1)
    # circuit.rzz(1.342, 0, 1)
    # circuit.h(1)
    # circuit.measure_all()
    provider = IBMQ.load_account()
    backend = provider.get_backend("simulator_statevector")
    qpu_runner = QPURunner({"sim": StatevectorSimulator()})
    decomposer = Decomposer(qpu_runner)
    
    job_id = decomposer.run(circuit, [], metadata=QVMJobMetadata(qpu_name="sim", shots=10000))
    quasi_distr = decomposer.get_results(job_id)[0]

    counts = quasi_distr.to_counts(10000)
    # print(counts)
    print(fidelity(circuit, counts))


if __name__ == "__main__":
    main()
