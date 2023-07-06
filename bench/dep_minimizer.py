import networkx as nx
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.reduce_deps import QubitDependencyMinimizer

from util.run import Benchmark, run_benchmark
from util._util import enable_logging
from circuits import get_circuits, BENCHMARKS


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

    NUM_VGATES = 3

    result_dir = f"bench/results/dep_min/{NUM_VGATES}"
    # service = QiskitRuntimeService()

    backend = FakeMontrealV2()

    # TODO: use this once we have access
    # backend = service.get_backend("ibmq_kolkata")
    runner = None

    for benchname in BENCHMARKS:
        circuits = get_circuits(benchname, (6, backend.num_qubits))
        bench_dep_min(
            f"{result_dir}/{benchname}.csv", circuits, backend, NUM_VGATES, runner
        )


if __name__ == "__main__":
    enable_logging()
    main()
