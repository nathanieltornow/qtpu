from benchmark import benchmark_QVM_layer

from qiskit.providers.fake_provider import FakeOslo
from qiskit.providers.ibmq import IBMQ, AccountProvider

from qvm.stack.decomposer import (
    BisectionDecomposer,
    QPUAwareDecomposer,
    LadderDecomposer,
)
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.simulator import SimulatorQPU


def scale_qpu_stack(provider: AccountProvider):
    qpu = SimulatorQPU(10)
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = LadderDecomposer(qpu_runner, 10)
    return stack


def main():
    benchmark_circuits = [
        # "hamiltonian/1_layer/10.qasm",
        # "hamiltonian/1_layer/12.qasm",
        # "hamiltonian/1_layer/14.qasm",
        # "hamiltonian/1_layer/16.qasm",
        # "hamiltonian/1_layer/18.qasm",
        # "hamiltonian/1_layer/20.qasm",
        "hamiltonian/1_layer/30.qasm",
        # "hamiltonian/1_layer/40.qasm",
        # "hamiltonian/1_layer/50.qasm",
        # "hamiltonian/1_layer/60.qasm",
        # "hamiltonian/1_layer/20.qasm",
        # "ghz/10.qasm",
        # "ghz/20.qasm",
        # "ghz/30.qasm",
    ] * 4
    benchmark_circuits = sorted(benchmark_circuits)
    provider = IBMQ.load_account()
    stack = scale_qpu_stack(provider)
    benchmark_QVM_layer(stack, benchmark_circuits, provider)


if __name__ == "__main__":
    main()
