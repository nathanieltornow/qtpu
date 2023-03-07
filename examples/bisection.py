import logging

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator


from qvm.cut_library.decomposition import bisect
from qvm.main import run_on_sim


logging.basicConfig(level=logging.INFO)


def main():
    num_qubits = 2
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

    circuit = bisect(circuit)
    circuit.draw(output="mpl", scale=0.7)
    print(circuit)

    quasi_distr = run_on_sim(circuit, 10000)

    actual_res = AerSimulator().run(circ_cp, shots=10000).result().get_counts()
    print(hellinger_fidelity(quasi_distr.to_counts(10000), actual_res))


if __name__ == "__main__":
    main()
