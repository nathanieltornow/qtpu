from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

from qvm import QVMCompiler
from qvm.compiler.virtualization import OptimalDecompositionPass
from qvm.compiler.distr_transpiler.backend_mapper import BasicBackendMapper

from .bench import RunConfiguration, Benchmark, IdentityCompiler, run_benchmark


RESULT_FILE = "bench/results/cutqc.csv"


def generate_cutqc_bench(
    circuits: list[QuantumCircuit],
    backend: BackendV2,
    size_to_reach: int,
    run_on_hardware: bool,
) -> Benchmark:
    bench = Benchmark(
        RESULT_FILE,
        circuits,
        RunConfiguration(
            compiler=QVMCompiler(
                virt_passes=[OptimalDecompositionPass(size_to_reach=size_to_reach)],
                dt_passes=[BasicBackendMapper(backend)],
            )
        ),
        RunConfiguration(compiler=IdentityCompiler()),
        run_on_hardware,
    )
    return bench


if __name__ == "__main__":
    pass
