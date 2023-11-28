from dataclasses import asdict, dataclass
import os
import csv

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

import qvm


@dataclass
class RunConfiguration:
    compiler: qvm.QVMCompiler
    shots: int = 20000
    optimization_level: int = 3


@dataclass
class Benchmark:
    result_file: str
    circuits: list[QuantumCircuit]
    run_config: RunConfiguration
    base_run_config: RunConfiguration | None = None


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
    pass


def _run_experiment(
    circuit: QuantumCircuit,
    run_config: RunConfiguration,
    base_run_config: RunConfiguration | None = None,
) -> BenchmarkResult:
    res = BenchmarkResult(num_qubits=circuit.num_qubits)

    vc = run_config.compiler.run(circuit)
    res.num_fragments = len(vc.fragment_circuits)
    res.num_instances = sum(len(insts) for insts in vc.instantiations().values())


def _virtual_circuit_stats(virtual_circuit: qvm.VirtualCircuit) -> tuple[int, int, int]:
    fragments = virtual_circuit.fragment_circuits.values()
    num_vgates = sum(
        len(frag.virtual_gates) for frag in virtual_circuit.fragment_circuits.values()
    )
    num_deps = sum(
        len(frag.virtual_gates) for frag in virtual_circuit.fragment_circuits.values()
    )
    num_cnots = sum(
        1
        for frag in virtual_circuit.fragment_circuits.values()
        for instr in frag
        if instr.operation.name == "cx"
    )
    depth = virtual_circuit.depth()
    return num_vgates, num_deps, num_cnots, depth
