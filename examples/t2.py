import logging

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.providers.ibmq import IBMQ
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

from qvm.bench import fidelity
from qvm.cut_library.decomposition import bisect
from qvm.runtime.qpus.ibmq import IBMQSimulator
from qvm.runtime.util import sample_on_ibmq_backend

provider = None


def get_circuit(num_qubits: int) -> QuantumCircuit:
    circuit = EfficientSU2(
        num_qubits=num_qubits,
        reps=2,
        entanglement="linear",
        su2_gates=["rx", "ry", "rz"],
    )
    circuit.measure_all()
    circuit = circuit.decompose()
    params = [(np.pi * i) / 16 for i in range(len(circuit.parameters))]
    circuit = circuit.bind_parameters(dict(zip(circuit.parameters, params)))
    return circuit


def main():
    circuit = get_circuit(10)

    backend = provider.get_backend("ibmq_qasm_simulator")

    vcircuit = bisect(circuit)

    quasi_distr = sample_on_ibmq_backend(vcircuit, backend, shots=100000)
    counts = quasi_distr.to_counts(100000)

    print(fidelity(circuit, counts))


if __name__ == "__main__":
    provider = IBMQ.load_account()
    main()
