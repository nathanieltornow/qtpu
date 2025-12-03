from mqt.bench import get_benchmark_indep

from qtpu import HEinsum
from qtpu.runtime import HEinsumRuntime
from qtpu.compiler import optimize, OptimizationParameters
from evaluation.benchmarks import get_benchmark


if __name__ == "__main__":

    circuit = get_benchmark("dist-vqe", circuit_size=20, cluster_size=10)

    heinsum = HEinsum.from_circuit(circuit)

    optimized_heinsum = optimize(heinsum, OptimizationParameters(seed=100)).select_best(
        cost_weight=1000, max_size=10
    )

    runtime = HEinsumRuntime(optimized_heinsum, backend="cudaq")
    runtime.prepare()
    result = runtime.execute()
    print(result)
