from qiskit.providers.fake_provider import FakeMontrealV2, FakeOslo
from qiskit.providers.ibmq import IBMQ, AccountProvider

from benchmark import benchmark_QVM_layer
from qvm.stack.decomposer import (BisectionDecomposer, LadderDecomposer,
                                  QPUAwareDecomposer)
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.ibmq_fake import IBMQFakeQPU
from qvm.stack.qpus.ibmq_qpu import IBMQQPU
from qvm.stack.qpus.simulator import SimulatorQPU


def scale_qpu_stack(provider: AccountProvider):
    # qpu = IBMQQPU(provider, "ibmq_qasm_simulator")
    qpu = IBMQFakeQPU(provider, "ibm_oslo")
    # qpu = SimulatorQPU(FakeOslo())
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = LadderDecomposer(qpu_runner, 4)
    return stack


def main():
    benchmark_circuits = [
        f"hamiltonian/2_{i}.qasm" for i in [4, 6, 8, 10, 12, 14, 16, 18, 20]
    ] * 4
    provider = IBMQ.load_account()
    stack = scale_qpu_stack(provider)
    benchmark_QVM_layer(stack, benchmark_circuits, provider)


if __name__ == "__main__":
    main()
