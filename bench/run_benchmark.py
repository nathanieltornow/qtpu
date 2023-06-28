from dataclasses import asdict
from bench_generation import (
    Benchmark,
    generate_vqr_benchmarks,
    generate_gen_bisection_benchmarks,
)
from tqdm import tqdm

from qvm.virtualizer import Virtualizer

from _util import append_to_csv_file
from _run_experiment import get_circuit_properties


def run_benchmark_stats(benches: list[Benchmark]):
    progress = tqdm(total=sum(len(b.circuits) for b in benches))
    progress.set_description("Running Benchmarks")
    for bench in benches:
        for circ in bench.circuits:
            cut_circ = bench.virt_compiler.run(circ)
            virt = Virtualizer(cut_circ)
            bench_stat = get_circuit_properties(circ, virt, bench.backend)
            append_to_csv_file(bench.result_file, asdict(bench_stat))
            progress.update(1)


if __name__ == "__main__":
    from qiskit.providers.fake_provider import FakeMontrealV2

    backend = FakeMontrealV2()

    # benches = generate_gen_bisection_benchmarks(
    #     "qaoa", [.1, .2, .3], backend, num_vgates=4, reverse_order=False
    # )
    benches = generate_gen_bisection_benchmarks(
        "vqe", [1, 2, 3, 4], backend, num_vgates=4, reverse_order=False
    )

    run_benchmark_stats(benches)
