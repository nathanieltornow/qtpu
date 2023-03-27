from benchmark import benchmark_QVM_layer

from qiskit.providers.fake_provider import FakeMontrealV2
from qiskit.providers.ibmq import IBMQ, AccountProvider

from qvm.stack.decomposer import BisectionDecomposer, QPUAwareDecomposer, LadderDecomposer
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.simulator import SimulatorQPU
from qvm.stack.qpus.ibmq_qpu import IBMQQPU
from qvm.stack.qpus.ibmq_fake import IBMQFakeQPU


def scale_qpu_stack(provider: AccountProvider):
    # qpu = IBMQQPU(provider, "ibmq_qasm_simulator")
    qpu = SimulatorQPU(FakeMontrealV2())
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = LadderDecomposer(qpu_runner, 4)
    return stack


def main():
    benchmark_circuits = [
        # "hamiltonian/1_layer/4.qasm",
        # "hamiltonian/1_layer/6.qasm",
        "hamiltonian/1_layer/8.qasm",
        # "hamiltonian/1_layer/10.qasm",
        # "hamiltonian/1_layer/12.qasm",
        # "hamiltonian/1_layer/14.qasm",
        # "hamiltonian/1_layer/16.qasm",
        # "hamiltonian/1_layer/18.qasm",
        # "hamiltonian/1_layer/20.qasm",
        # "qft/4.qasm",
    ] * 4
    benchmark_circuits = sorted(benchmark_circuits)
    provider = IBMQ.load_account()
    stack = scale_qpu_stack(provider)
    benchmark_QVM_layer(stack, benchmark_circuits, provider)


if __name__ == "__main__":
    main()
