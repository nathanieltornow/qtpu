import os
import csv
from dataclasses import dataclass, asdict
from time import perf_counter
from multiprocessing.pool import ThreadPool, Pool
from random import randint, random

import numpy as np

from qvm import QuasiDistr


@dataclass
class Benchmark:
    result_file: str
    num_qubits: int
    num_fragments: int
    num_vgates: int
    num_threads: int
    num_shots: int = 1000


@dataclass
class BenchmarkResult:
    num_qubits: int
    num_fragments: int
    num_vgates: int
    num_threads: int
    time: float

    def append_dict_to_csv(self, filepath: str) -> None:
        data = asdict(self)
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as csv_file:
                csv.DictWriter(csv_file, fieldnames=data.keys()).writeheader()
                csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)
            return

        with open(filepath, "a") as csv_file:
            csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)


def run_bench(benchmark: Benchmark) -> None:
    print(f"Generating workload")
    workload = _generate_knit_workload(
        num_fragments=benchmark.num_fragments,
        num_qubits=benchmark.num_qubits,
        num_vgates=benchmark.num_vgates,
        num_shots=benchmark.num_shots,
    )
    num_threads = min(benchmark.num_threads, workload.shape[1])

    print(f"Running benchmark with {num_threads} threads")
    print(f"Workload shape: {workload.shape}")

    split = np.array_split(workload, num_threads, axis=1)
    with Pool(num_threads) as pool:
        start = perf_counter()
        results = pool.map(_merge_and_knit, split)
        _ = np.sum(results)
        end = perf_counter()

    t = end - start

    res = BenchmarkResult(
        num_qubits=benchmark.num_qubits,
        num_fragments=benchmark.num_fragments,
        num_vgates=benchmark.num_vgates,
        num_threads=benchmark.num_threads,
        time=t,
    )
    res.append_dict_to_csv(benchmark.result_file)


def _generate_quasidistr(num_qubits: int, num_shots: int) -> QuasiDistr:
    qd = QuasiDistr({})
    for _ in range(num_shots):
        key = randint(0, 2**num_qubits - 1)
        if key in qd:
            qd[key] += random() * 0.01
        else:
            qd[key] = random() * 0.01
    return qd


def _generate_knit_workload(
    num_fragments: int, num_vgates: int, num_qubits: int, num_shots: int
) -> np.ndarray:
    qubits_per_fragment = num_qubits // num_fragments
    results = [
        _generate_quasidistr(qubits_per_fragment, num_shots)
        for _ in range(num_fragments * 6**num_vgates)
    ] + [1 / 2] * (6**num_vgates)

    workload = np.array(results, dtype=object)

    return workload.reshape((num_fragments + 1, 6**num_vgates))


def _merge_and_knit(results: np.ndarray) -> QuasiDistr:
    merged_results = np.prod(results, axis=0)
    return np.sum(merged_results, axis=0)


if __name__ == "__main__":
    for num_fragments, num_vgates in [(2, 1), (3, 2), (4, 3), (5, 4)]:
        for procs in [1, 2, 4, 8, 16, 32, 64]:
            bench = Benchmark(
                    f"bench/results/knit_hs1.csv",
                num_qubits=70,
                num_fragments=num_fragments,
                num_vgates=num_vgates,
                num_threads=procs,
                num_shots=5000,
            )
            run_bench(bench)
