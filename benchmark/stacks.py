from qiskit.providers.fake_provider import FakeMontrealV2, FakeOslo
from qiskit.providers.ibmq import IBMQ, AccountProvider

from qvm.stack.decomposer import (BisectionDecomposer, LadderDecomposer,
                                  QPUAwareDecomposer)
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.ibmq_fake import IBMQFakeQPU
from qvm.stack.qpus.ibmq_qpu import IBMQQPU
from qvm.stack.qpus.simulator import SimulatorQPU


def scale_fidelity_stack(provider: AccountProvider, num_qubits: int):
    # qpu = IBMQQPU(provider, "ibm_oslo")
    qpu = IBMQFakeQPU(provider, "ibm_oslo")
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = LadderDecomposer(qpu_runner, num_qubits)
    return stack


def scale_time_stack(num_qubits: int):
    qpu = SimulatorQPU()
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = LadderDecomposer(qpu_runner, num_qubits)
    return stack
