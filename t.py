import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister, Qubit
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit import transpile


from qvm.compiler.compiler import compile_circuit
from qvm.compiler.optimizer import NumQubitsOptimizer
from qvm.runtime.runtime import contract_hybrid_tn
from qvm.runtime.qpu_manager import SimulatorQPUManager


def measure_all(qc: QuantumCircuit) -> None:
    qc.add_register(ClassicalRegister(qc.num_qubits, name="c"))
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))


if __name__ == "__main__":

    qc = EfficientSU2(5, reps=1).decompose()
    qc = qc.assign_parameters({param: np.random.randn() / 2 for param in qc.parameters})

    # qc = QuantumCircuit(3)
    # qc.h(0)
    # qc.cx(0, 1)
    # qc.cx(1, 2)

    act_val = Estimator().run(qc, "Z" * qc.num_qubits).result().values

    measure_all(qc)

    hybrid_tn = compile_circuit(qc, NumQubitsOptimizer(2), {})

    for qtens in hybrid_tn.quantum_tensors:
        print(qtens._circuit)

    # cg = CircuitGraph(qc)

    # from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection

    # compressed = compress_qubits(cg)

    # print(len(compressed.nodes))

    # A, B = kernighan_lin_bisection(compressed)

    # # C, D = kernighan_lin_bisection(compressed.subgraph(A))
    # # # A, B = cut_qubits(cg, qc.qubits[:2], qc.qubits[2:])

    # hybrid_tn = cg.hybrid_tn(
    #     [decompress_nodes(A), decompress_nodes(B)]
    # )

    # for qt in hybrid_tn._quantum_tensors:
    #     print(qt._circuit)

    print(act_val)
    print(contract_hybrid_tn(hybrid_tn, SimulatorQPUManager()))
