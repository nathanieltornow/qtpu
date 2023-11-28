from dataclasses import asdict, dataclass
import os
import csv

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qvm.compiler import QVMCompiler


@dataclass
class Benchmark:
    result_file: str
    circuits: list[QuantumCircuit]
    compiler: QVMCompiler


@dataclass
class BenchmarkResult:
    num_qubits: int
    h_fid: float = np.nan
    h_fid_base: float = np.nan
    tv_fid: float = np.nan
    tv_fid_base: float = np.nan
    num_cnots: int = np.nan
    num_cnots_base: int = np.nan
    depth: int = np.nan
    depth_base: int = np.nan
    num_deps: int = np.nan
    num_deps_base: int = np.nan
    num_vgates: int = np.nan
    num_fragments: int = np.nan
    num_instances: int = np.nan
    run_time: float = np.nan
    knit_time: float = np.nan
    run_time_base: float = np.nan

    def append_dict_to_csv(self, filepath: str, data: dict) -> None:
        data = asdict(self)
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as csv_file:
                csv.DictWriter(csv_file, fieldnames=data.keys()).writeheader()
                csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)
            return

        with open(filepath, "a") as csv_file:
            csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)


def run_benchmark(bench: Benchmark) -> None:
    progress = tqdm(total=len(bench.circuits))
    progress.set_description("Running Bench Circuits")

    for circ in bench.circuits:
        virt = bench.compiler.run(circ)

        if bench.base_compiler is not None:
            base_virt = bench.base_compiler.run(circ)
        else:
            base_virt = VirtualCircuit(circ)

        res = _run_experiment(
            circ,
            virt,
            base_virt,
            bench.backend,
            runner=runner,
            base_backend=bench.base_backend,
        )
        res.append_to_csv(bench.result_file)
        progress.update(1)
