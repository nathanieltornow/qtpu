from benchmark import provider, run_on_QVM_layer
from qvm.stack.decomposer import BisectionDecomposer
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.simulator import IBMQSimulator


def scale_qpu_stack():
    qpu = IBMQSimulator(provider)
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = BisectionDecomposer(qpu_runner, 2)
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

    stack = scale_qpu_stack()
    run_on_QVM_layer(stack, benchmark_circuits)


if __name__ == "__main__":
    main()
