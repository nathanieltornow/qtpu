import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister, Qubit
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.primitives import Estimator
from qiskit import transpile
from qiskit_aer import AerSimulator

from qvm.compiler.oracle import MaxQubitsOracle
from qvm.compiler import compile


def measure_all(qc: QuantumCircuit) -> None:
    qc.add_register(ClassicalRegister(qc.num_qubits, name="c"))
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))


if __name__ == "__main__":

    qc = TwoLocal(20, ["ry", "rz"], "cx", "circular", reps=3).decompose()
    # qc = qc.assign_parameters({param: np.random.randn() / 2 for param in qc.parameters})

    # qc = QuantumCircuit(2)
    # qc.h(0)
    # qc.rzz(1.2, 0, 1)
    # qc.h(1)

    htn = compile(qc, oracle=MaxQubitsOracle(15), show_progress_bar=True)

    for qtn in htn.quantum_tensors:
        print(qtn.circuit)
