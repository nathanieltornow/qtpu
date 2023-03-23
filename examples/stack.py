import logging

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.providers.fake_provider import FakeOslo

from qiskit.providers.ibmq import IBMQ
from qiskit.compiler import transpile

# from multiprocessing.pool import Pool

from qvm.runtime.util import sample_on_ibmq_backend
from qvm.bench import fidelity

from qvm.stack._types import QVMJobMetadata
from qvm.stack.decomposer import Decomposer
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.simulator import IBMQSimulator


def get_circuit(num_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    circuit.cx(list(range(num_qubits - 1)), list(range(1, num_qubits)))
    circuit.measure_all()
    return circuit


def main():
    circuit = get_circuit(32)
    # circuit = QuantumCircuit(2)
    # circuit.h(0)
    # circuit.h(1)
    # circuit.rzz(1.342, 0, 1)
    # circuit.h(1)
    # circuit.measure_all()

    provider = IBMQ.load_account()
    qpu = IBMQSimulator(provider)
    
    qpu_runner = QPURunner({"sim": qpu})
    decomposer = Decomposer(qpu_runner, 2)

    job_id = decomposer.run(
        circuit, [], metadata=QVMJobMetadata(qpu_name="sim", shots=10000)
    )
    
    from  multiprocessing.pool import Pool
    
    with Pool() as pool:
        quasi_distr = decomposer.get_results(job_id, pool)[0]


    print(decomposer._stats)
    counts = quasi_distr.to_counts(10000)
    # print(counts)
    print(fidelity(circuit, counts, provider))


if __name__ == "__main__":
    main()
