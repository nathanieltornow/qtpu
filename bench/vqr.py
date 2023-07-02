from qiskit.providers import BackendV2

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.reduce_swap import ReduceSWAPCompiler

from util.circuits import two_local, qaoa
from util.run import Benchmark, run_benchmark


def bench_vqr_two_local(
    layers: int, backend: BackendV2, runner: QVMBackendRunner | None = None
) -> None:
    benchmark = Benchmark(
        circuits=[two_local(i, layers) for i in range(4, backend.num_qubits, 2)],
        backend=backend,
        result_file=f"bench/results/vqr/{backend.name}/2local_{layers}.csv",
        virt_compiler=ReduceSWAPCompiler(backend, 2, True),
    )
    run_benchmark(benchmark, runner)


def bench_vqr_qaoa(
    deg: int, backend: BackendV2, runner: QVMBackendRunner | None = None
) -> None:
    benchmark = Benchmark(
        circuits=[qaoa(i, deg) for i in range(4, backend.num_qubits, 2)],
        backend=backend,
        result_file=f"bench/results/vqr/{backend.name}/qaoa_{deg}.csv",
        virt_compiler=ReduceSWAPCompiler(backend, 2, False),
    )
    run_benchmark(benchmark, runner)


if __name__ == "__main__":
    from qiskit_ibm_runtime import QiskitRuntimeService

    from qiskit.providers.fake_provider import FakeMontrealV2

    # service = QiskitRuntimeService()

    # backend = service.get_backend("ibmq_kolkata")

    # runner = IBMBackendRunner(service=service)

    bench_vqr_two_local(1, FakeMontrealV2())
    # bench_vqr_qaoa(2, backend)
