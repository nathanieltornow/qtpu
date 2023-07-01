from qiskit.providers import BackendV2

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.gate_decomp import OptimalDecompositionCompiler

from bench_util.circuits import hamsim, qaoa, vqe
from bench_util.run import Benchmark, run_benchmark


def bench_hamsim(
    layers: int,
    backend: BackendV2,
    base_backend: BackendV2,
    runner: QVMBackendRunner | None = None,
) -> None:
    benchmark = Benchmark(
        circuits=[hamsim(i, layers) for i in range(2, 15, 2)],
        backend=backend,
        base_backend=base_backend,
        result_file=f"results/vqr/{backend.name}_{base_backend}/hamsim_{layers}.csv",
        virt_compiler=OptimalDecompositionCompiler(5),
    )
    run_benchmark(benchmark, runner)
    

def bench_vqe(
    layers: int,
    backend: BackendV2,
    base_backend: BackendV2,
    runner: QVMBackendRunner | None = None,
) -> None:
    benchmark = Benchmark(
        circuits=[hamsim(i, layers) for i in range(2, 15, 2)],
        backend=backend,
        base_backend=base_backend,
        result_file=f"results/noisy_scale/{backend.name}_{base_backend}/vqe_{layers}.csv",
        virt_compiler=OptimalDecompositionCompiler(5),
    )
    run_benchmark(benchmark, runner)


if __name__ == "__main__":
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()

    backend = service.get_backend("ibm_perth")
    base_backend = service.get_backend("ibmq_guadalupe")

    runner = IBMBackendRunner(service=service)

    bench_vqe(3, backend, base_backend, runner)
    # bench_vqr_qaoa(2, backend)
