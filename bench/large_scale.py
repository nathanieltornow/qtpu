from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

from qvm import QVMCompiler
from qvm.compiler.virtualization import OptimalDecompositionPass
from qvm.compiler.distr_transpiler.backend_mapper import BasicBackendMapper

from bench import RunConfiguration, Benchmark, IdentityCompiler, run_benchmark
from circuits.circuits import get_circuits


def generate_large_scale_bench(
    result_file: str,
    circuits: list[QuantumCircuit],
    backend: BackendV2,
    size_to_reach: int,
    budget: int,
) -> Benchmark:
    bench = Benchmark(
        result_file,
        circuits,
        RunConfiguration(
            compiler=QVMCompiler(
                virt_passes=[OptimalDecompositionPass(size_to_reach=size_to_reach)],
                dt_passes=[BasicBackendMapper(backend)],
            ),
            budget=budget,
        ),
        RunConfiguration(compiler=IdentityCompiler(backend)),
    )
    return bench


if __name__ == "__main__":
    circuits = get_circuits("vqe_2", (20, 101))

    from qiskit.providers.fake_provider import FakeSherbrooke

    bench = generate_large_scale_bench(
        "bench/results/large_scale.csv",
        circuits,
        FakeSherbrooke(),
        size_to_reach=5,
        budget=10,
    )

    run_benchmark(bench)
