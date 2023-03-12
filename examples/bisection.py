import logging

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.providers.ibmq import IBMQ
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

from qvm.cut_library.decomposition import bisect
from qvm.runtime.qpus.ibmq import IBMQSimulator
from qvm.runtime.runtime import QVMRuntime

logging.basicConfig(level=logging.INFO)

provider = None


def main():
    num_qubits = 6
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
    circ_cp = circuit.copy()

    runtime = QVMRuntime({IBMQSimulator(provider)})

    quasi_distr = runtime.sample(circuit)

    actual_res = AerSimulator().run(circ_cp, shots=10000).result().get_counts()
    print(hellinger_fidelity(quasi_distr.to_counts(10000), actual_res))


if __name__ == "__main__":
    provider = IBMQ.load_account()
    main()
