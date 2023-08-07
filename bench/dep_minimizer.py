import networkx as nx
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler import QVMCompiler
from qvm.compiler.virtualization.reduce_deps import (
    GreedyDependencyBreaker,
    CircularDependencyBreaker,
    QubitDependencyMinimizer,
)

from util.run import Benchmark, run_benchmark
from util._util import enable_logging
from circuits import get_circuits, BENCHMARK_CIRCUITS


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
        compiler=QVMCompiler(GreedyDependencyBreaker(num_vgates)),
    )
    run_benchmark(benchmark, runner)


def main():
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_provider import IBMProvider

    provider = IBMProvider(instance="ibm-q-unibw/reservations/reservations")

    NUM_VGATES = 1

    small_qpu = "ibmq_kolkata"
    large_qpu = "ibmq_kolkata"
    result_dir = f"bench/kolkata_results/dep_min/{NUM_VGATES}"

    backend = provider.get_backend(small_qpu)
    base_backend = provider.get_backend(large_qpu)

    provider2 = IBMProvider(instance="ibm-q-unibw/training/ht2022")

    runner = IBMBackendRunner(provider2, simulate_qpus=False)

    for benchname in [
        # "bv",
        # "vqe_1",
        # "vqe_2",
        # "vqe_3",
        # "hamsim_1",
        # "hamsim_2",
        # "twolocal_1",
        # "twolocal_2",
        "twolocal_3",
        "qaoa_r2",
        # "qaoa_r3"
        # "twolocal_3",
        # "hamsim_3",
        "qaoa_b",
        "vqe_1",
        "vqe_2",
        # "vqe_3",
        # "qaoa_ba2",
        # "qaoa_ba3",
    ]:
        circuits = (
            get_circuits(benchname, (6, 7))
            + get_circuits(benchname, (8, 10))
            + get_circuits(benchname, (10, 11))
            + get_circuits(benchname, (12, 13))
            # + get_circuits(benchname, (24, 25))
        )
        bench_dep_min(
            f"{result_dir}/{benchname}.csv", circuits, backend, NUM_VGATES, runner
        )


if __name__ == "__main__":
    enable_logging()
    main()
