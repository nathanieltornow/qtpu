import networkx as nx
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.reduce_swap import ReduceSWAPCompiler

from util.run import Benchmark, run_benchmark
from util.circuits import two_local, qaoa, qft, dj
from util._util import enable_logging


def bench_reduce_swap(
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
        virt_compiler=ReduceSWAPCompiler(backend, num_vgates),
    )
    run_benchmark(benchmark, runner)


def main():
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.providers.fake_provider import FakeMontrealV2

    result_dir = f"bench/results/swap_reduce"
    service = QiskitRuntimeService()

    # backend = FakeMontrealV2()

    # TODO: use this once we have access
    backend = service.get_backend("ibm_algiers")
    # runner = IBMBackendRunner(service)
    # runner = LocalBackendRunner()
    runner = None

    # for layer in range(1, 4):
    #     circuits = [two_local(i, layer) for i in range(4, backend.num_qubits, 2)]
    #     bench_reduce_swap(
    #         f"{result_dir}/2local_{layer}.csv", circuits, backend, layer, runner
    #     )

    for degree in range(2, 4):
        circuits = [qft(i, degree) for i in range(4, backend.num_qubits, 2)]
        bench_reduce_swap(
            f"{result_dir}/qft_{degree}.csv", circuits, backend, 3, runner
        )

    # for degree in range(2, 4):
    circuits = [dj(i) for i in range(4, backend.num_qubits, 2)]
    bench_reduce_swap(f"{result_dir}/dj.csv", circuits, backend, 3, runner)

    # for degree in range(2, 4):
    #     circuits = [
    #         qaoa(nx.random_regular_graph(degree, i))
    #         for i in range(4, backend.num_qubits, 2)
    #     ]
    #     bench_reduce_swap(
    #         f"{result_dir}/qaoa_r{degree}.csv", circuits, backend, degree, runner
    #     )

    # for degree in range(2, 4):
    #     circuits = [
    #         qaoa(nx.barabasi_albert_graph(i, degree))
    #         for i in range(4, backend.num_qubits, 2)
    #     ]
    #     bench_reduce_swap(
    #         f"{result_dir}/qaoa_ba{degree}.csv", circuits, backend, degree, runner
    #     )


if __name__ == "__main__":
    enable_logging()
    main()
