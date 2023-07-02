from qiskit.providers import BackendV2

from qiskit.circuit import QuantumCircuit
from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.gate_decomp import OptimalGateDecomposer

from util.circuits import hamsim, qaoa, vqe, two_local, ghz
from util.run import Benchmark, run_benchmark
from util._util import enable_logging


def bench_noisy_scale(
    result_file: str,
    circuits: list[QuantumCircuit],
    backend: BackendV2,
    base_backend: BackendV2,
    runner: QVMBackendRunner | None = None,
    fragment_size: int | None = None,
) -> None:
    if fragment_size is None:
        fragment_size = backend.num_qubits
    benchmark = Benchmark(
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        result_file=result_file,
        virt_compiler=OptimalGateDecomposer(fragment_size),
    )
    run_benchmark(benchmark, runner)


def main() -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()

    small_qpu = "ibm_perth"
    large_qpu = "ibmq_guadalupe"
    result_dir = f"bench/results/noisy_scale/{small_qpu}_vs_{large_qpu}"

    backend = service.get_backend(small_qpu)
    base_backend = service.get_backend(large_qpu)

    # runner = IBMBackendRunner(service=service)
    runner = LocalBackendRunner()

    # ghz
    circuits = [ghz(i) for i in range(4, 17, 2)]
    bench_noisy_scale(
        result_file=f"{result_dir}/ghz.csv",
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        runner=runner,
        fragment_size=4,
    )

    # hamsim
    for layer, fragsize in zip(range(1, 4), [4, 7, 7]):
        circuits = [hamsim(i, layer) for i in range(4, 17, 2)]
        bench_noisy_scale(
            result_file=f"{result_dir}/hamsim_{layer}.csv",
            circuits=circuits,
            backend=backend,
            base_backend=base_backend,
            runner=runner,
            fragment_size=fragsize,
        )

    # vqe
    for layer, fragsize in zip(range(1, 4), [4, 7, 7]):
        circuits = [vqe(i, layer) for i in range(4, 17, 2)]
        bench_noisy_scale(
            result_file=f"{result_dir}/vqe_{layer}.csv",
            circuits=circuits,
            backend=backend,
            base_backend=base_backend,
            runner=runner,
            fragment_size=fragsize,
        )

    # 2local
    circuits = [two_local(i, 1) for i in range(4, 15, 2)]
    bench_noisy_scale(
        result_file=f"{result_dir}/2local_1.csv",
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        runner=runner,
        fragment_size=7,
    )

    # qaoa
    circuits = [qaoa(i, 1) for i in range(4, 15, 2)]
    bench_noisy_scale(
        result_file=f"{result_dir}/qaoa_1.csv",
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        runner=runner,
        fragment_size=7,
    )


if __name__ == "__main__":
    enable_logging()
    main()
