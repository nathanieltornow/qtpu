import networkx as nx
from qiskit.providers import BackendV2
from qiskit.circuit import QuantumCircuit

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.gate_decomp import OptimalGateDecomposer
from qvm.compiler import QVMCompiler

from circuits import BENCHMARK_CIRCUITS, get_circuits
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
        compiler=QVMCompiler(OptimalGateDecomposer(fragment_size)),
    )
    run_benchmark(benchmark, runner)


def main() -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.providers.fake_provider import FakeGuadalupeV2, FakePerth

    service = QiskitRuntimeService()

    small_qpu = "ibm_perth"
    large_qpu = "ibmq_lima"
    result_dir = f"bench/results/noisy_scale_real/{small_qpu}_vs_{large_qpu}"

    # backend = FakePerth()
    # base_backend = FakeGuadalupeV2()

    backend = service.get_backend(small_qpu)
    base_backend = service.get_backend(large_qpu)

    runner = IBMBackendRunner(service=service, simulate_qpus=False)
    # runner = LocalBackendRunner()
    # runner = None

    # for benchname in BENCHMARK_CIRCUITS:
    #     circuits = get_circuits(benchname, (6, 17))

    # benchname = "vqe_1"
    # circuits = (
    #     # get_circuits(benchname, (6, 7))
    #     # + get_circuits(benchname, (10, 11))
    #     get_circuits(benchname, (12, 13))
    # )
    # bench_noisy_scale(
    #     result_file=f"{result_dir}/{benchname}.csv",
    #     circuits=circuits,
    #     backend=backend,
    #     base_backend=base_backend,
    #     runner=runner,
    #     fragment_size=4,
    # )

    # benchname = "vqe_2"
    # circuits = (
    #     get_circuits(benchname, (6, 7))
    #     + get_circuits(benchname, (10, 11))
    #     + get_circuits(benchname, (12, 13))
    # )
    # bench_noisy_scale(
    #     result_file=f"{result_dir}/{benchname}.csv",
    #     circuits=circuits,
    #     backend=backend,
    #     base_backend=base_backend,
    #     runner=runner,
    #     fragment_size=7,
    # )

    # benchname = "vqe_3"
    # circuits = (
    #     get_circuits(benchname, (6, 7))
    #     + get_circuits(benchname, (10, 11))
    #     + get_circuits(benchname, (14, 15))
    # )
    # bench_noisy_scale(
    #     result_file=f"{result_dir}/{benchname}.csv",
    #     circuits=circuits,
    #     backend=backend,
    #     base_backend=base_backend,
    #     runner=runner,
    #     fragment_size=7,
    # )

    benchname = "qaoa_b"
    circuits = (
        get_circuits(benchname, (6, 7))
        + get_circuits(benchname, (10, 11))
        + get_circuits(benchname, (12, 13))
    )
    bench_noisy_scale(
        result_file=f"{result_dir}/{benchname}.csv",
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        runner=runner,
        fragment_size=7,
    )

    benchname = "hamsim_1"
    circuits = (
        get_circuits(benchname, (6, 7))
        + get_circuits(benchname, (10, 11))
        + get_circuits(benchname, (12, 13))
    )
    bench_noisy_scale(
        result_file=f"{result_dir}/{benchname}.csv",
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        runner=runner,
        fragment_size=4,
    )

    benchname = "hamsim_2"
    circuits = (
        get_circuits(benchname, (6, 7))
        + get_circuits(benchname, (10, 11))
        + get_circuits(benchname, (12, 13))
    )
    bench_noisy_scale(
        result_file=f"{result_dir}/{benchname}.csv",
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        runner=runner,
        fragment_size=7,
    )

    benchname = "ghz"
    circuits = (
        # get_circuits(benchname, (6, 7))
        # + get_circuits(benchname, (10, 11))
        get_circuits(benchname, (12, 13))
    )
    bench_noisy_scale(
        result_file=f"{result_dir}/{benchname}.csv",
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        runner=runner,
        fragment_size=4,
    )

    benchname = "wstate"
    circuits = (
        get_circuits(benchname, (6, 7))
        + get_circuits(benchname, (10, 11))
        + get_circuits(benchname, (12, 15))
    )
    bench_noisy_scale(
        result_file=f"{result_dir}/{benchname}.csv",
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        runner=runner,
        fragment_size=7,
    )

    benchname = "qsvm"
    circuits = (
        get_circuits(benchname, (6, 7))
        + get_circuits(benchname, (10, 11))
        + get_circuits(benchname, (12, 13))
    )
    bench_noisy_scale(
        result_file=f"{result_dir}/{benchname}.csv",
        circuits=circuits,
        backend=backend,
        base_backend=base_backend,
        runner=runner,
        fragment_size=7,
    )

    # benchname = "hamsim_3"
    # circuits = (
    #     get_circuits(benchname, (6, 7))
    #     + get_circuits(benchname, (10, 11))
    #     + get_circuits(benchname, (14, 15))
    # )
    # bench_noisy_scale(
    #     result_file=f"{result_dir}/{benchname}.csv",
    #     circuits=circuits,
    #     backend=backend,
    #     base_backend=base_backend,
    #     runner=runner,
    #     fragment_size=7,
    # )


if __name__ == "__main__":
    enable_logging()
    main()
