from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

from qvm import QVMCompiler
from qvm.compiler.virtualization import OptimalDecompositionPass
from qvm.compiler.distr_transpiler.backend_mapper import BasicBackendMapper

from bench import RunConfiguration, Benchmark, IdentityCompiler, run_benchmark
from circuits.circuits import get_circuits


RESULT_FILE = "bench/results/parallelism"


def generate_parallelism_bench(
    circuits: list[QuantumCircuit],
    size_to_reach: int,
    num_processes: int = 1,
):
    bench = Benchmark(
        RESULT_FILE + "_" + str(size_to_reach) + " " + str(num_processes) + ".csv",
        circuits,
        run_config=RunConfiguration(
            compiler=QVMCompiler(virt_passes=[OptimalDecompositionPass(size_to_reach)]),
            budget=6,
            num_processes=num_processes,
        ),
        run_on_hardware=True,
    )
    return bench





if __name__ == "__main__":
    circuits = get_circuits("hamsim_1", (20, 21))
    run_benchmark(generate_parallelism_bench(circuits, 5, 1))
    run_benchmark(generate_parallelism_bench(circuits, 5, 2))
    run_benchmark(generate_parallelism_bench(circuits, 5, 4))
    run_benchmark(generate_parallelism_bench(circuits, 5, 8))
    run_benchmark(generate_parallelism_bench(circuits, 5, 16))
    # run_benchmark(generate_parallelism_bench(circuits, 5, 32))
