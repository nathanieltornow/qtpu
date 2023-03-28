from benchmark import benchmark_QVM_layer

from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import FakeOslo
from qiskit.providers.ibmq import IBMQ, AccountProvider

from qvm.stack.decomposer import (
    BisectionDecomposer,
    QPUAwareDecomposer,
    LadderDecomposer,
)
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.simulator import SimulatorQPU
from qvm.sampler import QVMSampler


def scale_qpu_stack():
    qpu = SimulatorQPU()
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = LadderDecomposer(qpu_runner, 4)
    return stack


def main():
    benchmark_circuits = [
        # "hamiltonian/1_layer/10.qasm",
        # "hamiltonian/1_layer/12.qasm",
        # "hamiltonian/1_layer/14.qasm",
        # "hamiltonian/1_layer/16.qasm",
        # "hamiltonian/1_layer/18.qasm",
        # "hamiltonian/1_layer/20.qasm",
        # "hamiltonian/1_layer/30.qasm",
        # "hamiltonian/1_layer/40.qasm",
        # "hamiltonian/1_layer/50.qasm",
        # "hamiltonian/1_layer/60.qasm",
        # "hamiltonian/1_layer/20.qasm",
        "ghz/4.qasm",
        "ghz/6.qasm",
        # "ghz/10.qasm",
        # "ghz/20.qasm",
        # "ghz/30.qasm",
    ]

    stack = scale_qpu_stack()

    sampler = QVMSampler(stack)
    results = sampler.run([QuantumCircuit.from_qasm_file(c) for c in benchmark_circuits])
    print(results.result())


if __name__ == "__main__":
    main()
