import networkx as nx
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.reduce_deps import QubitDependencyMinimizer

from util.run import Benchmark, run_benchmark
from util.circuits import two_local, qaoa
from util._util import enable_logging


def bench_dep_min(
    result_file: str,
    circuits: list[QuantumCircuit],
    backend: BackendV2,
    num_vgates: int,
    runner: QVMBackendRunner | None = None,
) -> None:
    benchmark = Benchmark(
        circuits=circuits,
        backend=backend,
        result_file=result_file,
        virt_compiler=QubitDependencyMinimizer(num_vgates),
    )
    run_benchmark(benchmark, runner)


def main():
    # from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.providers.fake_provider import FakeMontrealV2

    result_dir = f"bench/results/dep_min"
    # service = QiskitRuntimeService()

    backend = FakeMontrealV2()

    # TODO: use this once we have access
    # backend = service.get_backend("ibmq_kolkata")
    runner = None

    # for layer in range(1, 4):
    #     circuits = [two_local(i, layer) for i in range(4, backend.num_qubits, 2)]
    #     bench_dep_min(
    #         f"{result_dir}/2local_{layer}.csv", circuits, backend, layer, runner
    #     )

    # for degree in range(2, 4):
    #     circuits = [
    #         qaoa(nx.random_regular_graph(degree, i))
    #         for i in range(4, backend.num_qubits, 2)
    #     ]
    #     bench_dep_min(
    #         f"{result_dir}/qaoa_r{degree}.csv", circuits, backend, degree, runner
    #     )
        
    circuits = [qaoa(nx.barbell_graph(i, 0)) for i in range(2, backend.num_qubits//2, 1)]
    bench_dep_min(
        f"{result_dir}/qaoa_b.csv", circuits, backend, 2, runner   
    )

    exit(0)

    for degree in range(1, 4):
        circuits = [
            qaoa(nx.barabasi_albert_graph(i, degree))
            for i in range(4, backend.num_qubits, 2)
        ]
        bench_dep_min(
            f"{result_dir}/qaoa_ba{degree}.csv", circuits, backend, degree, runner
        )


if __name__ == "__main__":
    enable_logging()
    main()
