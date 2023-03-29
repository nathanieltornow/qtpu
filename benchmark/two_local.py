from qiskit.circuit import QuantumCircuit
import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.providers.ibmq import IBMQ, AccountProvider

from qvm.stack.decomposer import (
    BisectionDecomposer,
    QPUAwareDecomposer,
    LadderDecomposer,
)
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.simulator import SimulatorQPU
from qvm.stack.qpus.ibmq_qpu import IBMQQPU
from qvm.stack.qpus.ibmq_fake import IBMQFakeQPU


from benchmark import benchmark_QVM_layer

REPS = 2
NUMS_QUBITS = [12, 14, 18, 20] * 5
BENCHNAME = "2reps6"


def two_local_circuit(num_qubits: int, reps: int):
    num_qubits = num_qubits
    circuit = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement="linear",
        entanglement_blocks="rzz",
        reps=reps,
    )
    circuit.measure_all()
    circuit = circuit.decompose()
    params = [
        (np.pi * np.random.uniform(0.0, 1.0)) for _ in range(len(circuit.parameters))
    ]
    return circuit.bind_parameters(params)


def scale_qpu_stack(provider: AccountProvider):
    # qpu = IBMQQPU(provider, "ibmq_qasm_simulator")
    qpu = IBMQFakeQPU(provider, "ibm_oslo")
    # qpu = SimulatorQPU(FakeOslo())
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = LadderDecomposer(qpu_runner, 6)
    return stack


def main():
    bench_circuits = [two_local_circuit(num_qubits, REPS) for num_qubits in NUMS_QUBITS]
    bench_circuits = sorted(bench_circuits, key=lambda c: c.num_qubits)
    provider = IBMQ.load_account()
    stack = scale_qpu_stack(provider)
    benchmark_QVM_layer(
        stack, bench_circuits, provider, f"results/{BENCHNAME}_two_local.csv"
    )


if __name__ == "__main__":
    main()
