from benchmark import benchmark_QVM_layer

from qiskit.providers.fake_provider import FakeOslo
from qiskit.providers.ibmq import IBMQ, AccountProvider

from qvm.stack.decomposer import BisectionDecomposer, QPUAwareDecomposer
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.ibmq_qpu import IBMQQPU
from qvm.stack.qpus.ibmq_fake import IBMQFakeQPU


def scale_qpu_stack(provider: AccountProvider):
    # qpu = IBMQQPU(provider, "simulator_statevector")
    qpu = IBMQFakeQPU(provider, "ibm_oslo")
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = QPUAwareDecomposer(qpu_runner, 4)
    return stack


def main():
    benchmark_circuits = [
        "hamiltonian/1_layer/4.qasm",
        "hamiltonian/1_layer/6.qasm",
        "hamiltonian/1_layer/8.qasm",
        "hamiltonian/1_layer/10.qasm",
        "hamiltonian/1_layer/12.qasm",
        "hamiltonian/1_layer/14.qasm",
        "hamiltonian/1_layer/16.qasm",
        "hamiltonian/1_layer/18.qasm",
        "hamiltonian/1_layer/20.qasm",
    ]
    provider = IBMQ.load_account()
    stack = scale_qpu_stack(provider)
    benchmark_QVM_layer(stack, benchmark_circuits, provider)


if __name__ == "__main__":
    main()
