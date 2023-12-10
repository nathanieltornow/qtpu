from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

from qvm import QVMCompiler
from qvm.compiler.virtualization import BisectionPass
from qvm.compiler.distr_transpiler.backend_mapper import BasicBackendMapper

from bench import RunConfiguration, Benchmark, IdentityCompiler, run_benchmark
from circuits.circuits import get_circuits


def generate_large_scale_bench(
    result_file: str,
    circuits: list[QuantumCircuit],
    backend: BackendV2,
    size_to_reach: int,
    budget: int,
    compare_to_base: bool = True,
) -> Benchmark:
    bench = Benchmark(
        result_file,
        circuits,
        RunConfiguration(
            compiler=QVMCompiler(
                virt_passes=[BisectionPass(size_to_reach=size_to_reach)],
                dt_passes=[BasicBackendMapper(backend)],
            ),
            budget=budget,
        ),
        RunConfiguration(compiler=IdentityCompiler(backend))
        if compare_to_base
        else None,
    )
    return bench


def scale_virts():
    from qiskit.providers.fake_provider import FakeSherbrooke

    circuits = get_circuits("vqe_1", (20, 100)) * 3
    for budget in [0, 2, 4, 6, 8]:
        bench = generate_large_scale_bench(
            f"bench/results/large_scale_{budget}.csv",
            circuits,
            FakeSherbrooke(),
            size_to_reach=20,
            budget=budget,
            compare_to_base=False,
        )
        run_benchmark(bench)

    circuits = get_circuits("vqe_1", (100, 501)) * 3
    for budget in [0, 2, 4, 6, 8]:
        bench = generate_large_scale_bench(
            f"bench/results/large_scale_{budget}.csv",
            circuits,
            FakeSherbrooke(),
            size_to_reach=100,
            budget=budget,
            compare_to_base=False,
        )

        run_benchmark(bench)

    # bench = generate_large_scale_bench(
    #     "bench/results/large_scale.csv",
    #     circuits,
    #     FakeSherbrooke(),
    #     size_to_reach=100,
    #     budget=10,
    # )

    run_benchmark(bench)


if __name__ == "__main__":
    scale_virts()
