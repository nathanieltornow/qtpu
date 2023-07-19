import networkx as nx
from qiskit.providers import BackendV2
from qiskit.circuit import QuantumCircuit

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.reduce_deps import GreedyDependencyBreaker
from qvm.compiler.virtualization.gate_decomp import OptimalGateDecomposer
from qvm.compiler.qubit_reuse import QubitReuseCompiler
from qvm.compiler import QVMCompiler

from circuits import BENCHMARK_CIRCUITS, get_circuits
from util.run import Benchmark, run_benchmark
from util._util import enable_logging


def bench_cut_vs_reuse(
    result_file: str,
    circuits: list[QuantumCircuit],
    backend: BackendV2,
    num_vgates: int,
    runner: QVMBackendRunner | None = None,
    fragment_size: int | None = None,
) -> None:
    if fragment_size is None:
        fragment_size = backend.num_qubits
    benchmark = Benchmark(
        circuits=circuits,
        backend=backend,
        result_file=result_file,
        compiler=QVMCompiler(OptimalGateDecomposer(fragment_size)),
        base_compiler=QVMCompiler(None, QubitReuseCompiler(fragment_size)),
    )
    run_benchmark(benchmark, runner)


def main():
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.providers.fake_provider import FakeGuadalupeV2, FakePerth

    service = QiskitRuntimeService()

    small_qpu = "ibm_perth"
    result_dir = f"bench/results/reuse_vs_cut/{small_qpu}"

    # backend = FakePerth()
    # base_backend = FakeGuadalupeV2()

    backend = service.get_backend(small_qpu)

    runner = None

    for benchname in [
        "wstate",
        # "twolocal_2",
        # "twolocal_3",
        "hamsim_2",
        "vqe_2",
        # "qaoa_b",
        # "qaoa_r2",
        # "qaoa_r3",
        # "qaoa_r4",
    ]:
        circuits = get_circuits(benchname, (8, 11))
        bench_cut_vs_reuse(
            result_file=f"{result_dir}/{benchname}.csv",
            circuits=circuits,
            backend=backend,
            num_vgates=1,
            runner=runner,
            fragment_size=5,
        )


if __name__ == "__main__":
    main()
