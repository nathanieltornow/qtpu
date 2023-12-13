from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

from qvm.quasi_distr import QuasiDistr


def calculate_fidelity(circuit: QuantumCircuit, noisy_result: QuasiDistr) -> float:
    ideal_result = (
        AerSimulator()
        .run(transpile(circuit, AerSimulator(), optimization_level=0), shots=100000)
        .result()
        .get_counts()
    )
    from qvm.runtime.runner import z_expval

    print(z_expval(ideal_result))
    # return hellinger_fidelity(ideal_result, noisy_result)
