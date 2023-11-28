from dataclasses import asdict, dataclass
import os
import csv

from tqdm import tqdm
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit.providers import BackendV2
from qiskit.quantum_info import hellinger_fidelity

import qvm
from qvm.compiler.dag import DAG
from qvm.compiler.distr_transpiler.backend_mapper import BasicBackendMapper


class IdentityCompiler(qvm.QVMCompiler):
    """Compiler that only maps the only fragment to a given backend"""

    def __init__(self, backend: BackendV2) -> None:
        super().__init__(dt_passes=[BasicBackendMapper(backend)])


@dataclass
class RunConfiguration:
    compiler: qvm.QVMCompiler
    budget: int = 0
    shots: int = 20000
    optimization_level: int = 3


@dataclass
class Benchmark:
    result_file: str
    circuits: list[QuantumCircuit]
    run_config: RunConfiguration
    base_run_config: RunConfiguration | None = None
    run_on_hardware: bool = False


@dataclass
class BenchmarkResult:
    num_qubits: int
    fid: float = np.nan
    fid_base: float = np.nan
    esp: float = np.nan
    esp_base: float = np.nan
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


def run_benchmark(bench: Benchmark) -> None:
    progress = tqdm(total=len(bench.circuits))
    progress.set_description(f"Benchmarking for {bench.result_file}")
    for circuit in bench.circuits:
        br = _run_experiment(
            circuit,
            bench.run_config,
            bench.base_run_config,
            bench.run_on_hardware,
        )
        br.append_dict_to_csv(bench.result_file)
        progress.update(1)


def _run_experiment(
    circuit: QuantumCircuit,
    run_config: RunConfiguration,
    base_run_config: RunConfiguration | None = None,
    run_on_hardware: bool = False,
) -> BenchmarkResult:
    br = BenchmarkResult(num_qubits=circuit.num_qubits)

    # first, do the qvm run
    vc = run_config.compiler.run(circuit, budget=run_config.budget)
    br.num_fragments = len(vc.fragment_circuits)
    br.num_instances = sum(len(insts) for insts in vc.instantiations().values())
    br.num_cnots, br.depth, br.num_deps = _virtual_circuit_stats(vc)
    br.esp = _compute_esp(vc)

    if run_on_hardware:
        qvm_result, run_info = qvm.run(
            vc, shots=run_config.shots, optimization_level=run_config.optimization_level
        )
        br.fid = _compute_fidelity(circuit, qvm_result)
        br.run_time = run_info.qpu_time
        br.knit_time = run_info.knit_time

    if base_run_config is None:
        return br

    # now, do the base run if it exists

    vc_base = base_run_config.compiler.run(circuit, 0)
    br.num_cnots_base, br.depth_base, br.num_deps_base = _virtual_circuit_stats(vc_base)
    br.esp_base = _compute_esp(vc_base)

    if run_on_hardware:
        qvm_result_base, run_info_base = qvm.run(
            vc_base,
            shots=base_run_config.shots,
            optimization_level=base_run_config.optimization_level,
        )
        br.fid_base = _compute_fidelity(circuit, qvm_result_base)
        br.run_time_base = run_info_base.qpu_time

    return br


def _virtual_circuit_stats(virtual_circuit: qvm.VirtualCircuit) -> tuple[int, int, int]:
    num_deps = max(
        DAG(frag_circ).num_dependencies()
        for frag_circ in virtual_circuit.fragment_circuits.values()
    )
    fragments = [
        transpile(
            frag_circ,
            backend=virtual_circuit.metadata[frag].backend,
            optimization_level=3,
        )
        for frag, frag_circ in virtual_circuit.fragment_circuits.items()
    ]
    num_cnots = max(
        sum(1 for instr in frag if instr.operation.name == "cx") for frag in fragments
    )
    depth = max(frag.depth() for frag in fragments)
    return num_cnots, depth, num_deps


def _compute_fidelity(circuit: QuantumCircuit, noisy_result: qvm.QuasiDistr) -> float:
    ideal_result = qvm.QuasiDistr.from_counts(
        AerSimulator()
        .run(transpile(circuit, AerSimulator(), optimization_level=0), shots=20000)
        .result()
        .get_counts()
    )
    return hellinger_fidelity(ideal_result, noisy_result)


def _compute_esp(virtual_circuit: qvm.VirtualCircuit) -> float:
    return 0.0
