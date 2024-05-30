import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister, Qubit
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.primitives import Estimator
from qiskit import transpile
from qiskit_aer import AerSimulator

from qvm.compiler.oracle import NumQubitsOracle, LeafOracle
from qvm.compiler.compile import compile


def measure_all(qc: QuantumCircuit) -> None:
    qc.add_register(ClassicalRegister(qc.num_qubits, name="c"))
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))


if __name__ == "__main__":

    from bench.benchmarks import generate_benchmark

    N = 20

    qc = generate_benchmark("vqe_1", N)

    # qc = TwoLocal(30, ["ry", "rz"], "cx", "circular", reps=2).decompose()
    # qc = qc.assign_parameters({param: np.random.randn() / 2 for param in qc.parameters})

    # qc = QuantumCircuit(4)
    # qc.h(range(4))
    # qc.rzz(1.2, 0, 1)
    # # qc.h(1)
    # qc.cx(1, 2)
    # qc.cx(2, 3)
    # qc.h(range(1))
    # # qc.rzz(1.2, 0, 1)
    # # qc.h(1)
    # # qc.cx(1, 2)
    # # qc.cx(2, 3)
    # # qc.h(range(4))

    htn = compile(
        qc,
        # compress="qubits",
        oracle=NumQubitsOracle(N // 3 + 1),
        max_trials=100,
        show_progress_bar=True,
    )

    for qtn in htn.quantum_tensors:
        print(qtn.circuit)
