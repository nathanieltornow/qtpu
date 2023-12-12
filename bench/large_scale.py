from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
import numpy as np

from qvm import QVMCompiler
from qvm.compiler.virtualization import BisectionPass, OptimalDecompositionPass
from qvm.compiler.virtualization.gate_decomp import _decompose_qubit_sets
from qvm.compiler.types import VirtualizationPass
from qvm.compiler.dag import DAG
from qvm.compiler.distr_transpiler.backend_mapper import BasicBackendMapper

from bench import RunConfiguration, Benchmark, IdentityCompiler, run_benchmark
from circuits.circuits import get_circuits


class CustomOptimalDecompositionPass(VirtualizationPass):
    def __init__(self, num_fragments: int) -> None:
        self._num_fragments = num_fragments
        super().__init__()

    def run(self, circuit: QuantumCircuit, budget: int) -> QuantumCircuit:
        frag_size = int(np.ceil(circuit.num_qubits / self._num_fragments))
        # divide qubits into sets of size frag_size
        qubit_sets = [
            set(circuit.qubits[i * frag_size : (i + 1) * frag_size])
            for i in range(self._num_fragments)
        ]
        print(max(len(qs) for qs in qubit_sets))
        dag = DAG(circuit)
        _decompose_qubit_sets(dag, qubit_sets)
        dag.fragment()
        return dag.to_circuit()


def generate_large_scale_bench(
    result_file: str,
    circuits: list[QuantumCircuit],
    backend: BackendV2,
    num_fragments: int = 1,
    compare_to_base: bool = True,
) -> Benchmark:
    bench = Benchmark(
        result_file,
        circuits,
        RunConfiguration(
            compiler=QVMCompiler(
                virt_passes=[CustomOptimalDecompositionPass(num_fragments)],
                dt_passes=[BasicBackendMapper(backend)],
            ),
            budget=100,
        ),
        RunConfiguration(compiler=IdentityCompiler(backend))
        if compare_to_base
        else None,
    )
    return bench


from qiskit.providers.fake_provider import FakeSherbrooke


def _run(budget: int):
    layers = 1
    circuits = get_circuits(f"vqe_{layers}", (600, 1001))
    bench = generate_large_scale_bench(
        f"bench/results/large_scale_{budget}.csv",
        circuits,
        FakeSherbrooke(),
        num_fragments=budget // layers + 1,
        compare_to_base=False,
    )

    run_benchmark(bench)


def scale_virts():
    import multiprocessing as mp

    with mp.Pool(8) as pool:
        pool.map(_run, [0, 2, 4, 6, 8])

    # circuits = get_circuits("vqe_2", (20, 100)) * 3
    # for budget in [2, 4, 6, 8]:
    #     bench = generate_large_scale_bench(
    #         f"bench/results/large_scale_{budget}.csv",
    #         circuits,
    #         FakeSherbrooke(),
    #         size_to_reach=20,
    #         budget=budget,
    #         compare_to_base=False,
    #     )
    #     run_benchmark(bench)

    # circuits = get_circuits("vqe_1", (100, 501))
    # for budget in [0, 2, 4, 6, 8]:
    #     bench = generate_large_scale_bench(
    #         f"bench/results/large_scale_{budget}.csv",
    #         circuits,
    #         FakeSherbrooke(),
    #         num_fragments=budget // 2 + 1,
    #         compare_to_base=False,
    #     )

    #     run_benchmark(bench)


if __name__ == "__main__":
    scale_virts()
