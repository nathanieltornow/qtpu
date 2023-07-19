from qiskit.providers import BackendV2
from qiskit.circuit import QuantumCircuit

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.gate_decomp import OptimalGateDecomposer
from qvm.compiler import QVMCompiler

from circuits import get_circuits
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
    from qiskit_ibm_provider import IBMProvider

    provider = IBMProvider(instance="ibm-q-unibw/reservations/reservations")

    small_qpu = "ibmq_kolkata"
    large_qpu = "ibmq_kolkata"
    result_dir = f"bench/perth_results/noisy_scale/"

    backend = provider.get_backend(small_qpu)
    base_backend = provider.get_backend(large_qpu)

    provider2 = IBMProvider(instance="ibm-q-unibw/training/ht2022")

    runner = IBMBackendRunner(provider2, simulate_qpus=False)

    for benchname in [
        # "twolocal_1",
        # "hamsim_1",
        # "hamsim_2",
        # "vqe_1",
        # "vqe_2",
        # "wstate",
        "qsvm",
        "ghz",
        "qaoa_b",
        "qaoa_ba2",
    ]:
        circuits = (
            # get_circuits(benchname, (10, 11))
            get_circuits(benchname, (8, 9))
            + get_circuits(benchname, (10, 11))
            + get_circuits(benchname, (14, 15))
            # + get_circuits(benchname, (18, 19))
            # + get_circuits(benchname, (20, 21))
            # + get_circuits(benchname, (24, 25))
        )
        bench_noisy_scale(
            result_file=f"{result_dir}/{benchname}.csv",
            circuits=circuits,
            backend=backend,
            base_backend=base_backend,
            runner=runner,
            fragment_size=13,
        )


if __name__ == "__main__":
    enable_logging()
    main()
