import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister, Qubit
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit import transpile


from qvm.compiler.compiler import compile_circuit
from qvm.compiler.optimizer import NumQubitsOptimizer
from qvm.runtime.runtime import contract_hybrid_tn
from qvm.runtime.qpu_manager import (
    SimulatorQPUManager,
    _DummyQPUManager,
)

def measure_all(qc: QuantumCircuit) -> None:
    qc.add_register(ClassicalRegister(qc.num_qubits, name="c"))
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))


if __name__ == "__main__":

    qc = EfficientSU2(20, reps=1).decompose()
    qc = qc.assign_parameters({param: np.random.randn() / 2 for param in qc.parameters})

    qc.remove_final_measurements(inplace=True)

    act_val = Estimator().run(qc, "Z" * qc.num_qubits).result().values
    #
    measure_all(qc)

    hybrid_tn = compile_circuit(qc, NumQubitsOptimizer(10, "rm_1q"))

    for qtens in hybrid_tn.quantum_tensors:
        print(qtens._circuit)

    # print(act_val)
    from time import perf_counter

    import sys

    start = perf_counter()

    print(contract_hybrid_tn(hybrid_tn, SimulatorQPUManager(True), 200000))
    print(perf_counter() - start)
    print(act_val)
    
