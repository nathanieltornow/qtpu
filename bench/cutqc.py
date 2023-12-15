import os
import csv
from dataclasses import dataclass, asdict

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.compiler import transpile
from qiskit_ibm_provider import IBMProvider

import qvm
from qvm import QVMCompiler, QuasiDistr
from qvm.compiler.virtualization import OptimalDecompositionPass
from qvm.compiler.distr_transpiler.backend_mapper import BasicBackendMapper
from qvm.compiler.dag import DAG
from qiskit.providers.fake_provider import FakeKolkataV2

from circuit_knitting.cutting.cutqc import (
    cut_circuit_wires,
    evaluate_subcircuits,
    reconstruct_full_distribution,
)

from bench_main import (
    _compute_fidelity,
    _esp,
    _virtual_circuit_stats,
)
from circuits.circuits import get_circuits, vqe


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
    num_cuts_base: int = np.nan

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


def run_cutqc_benchmark(
    result_file: str,
    circuits: list[QuantumCircuit],
    backend: BackendV2,
    size_to_reach: int,
    budget: int = 4,
    run_on_hardware: bool = False,
) -> None:
    comp = QVMCompiler(
        virt_passes=[OptimalDecompositionPass(size_to_reach=size_to_reach)],
        dt_passes=[BasicBackendMapper(backend)],
    )
    for circuit in circuits:
        result = BenchmarkResult(num_qubits=circuit.num_qubits)

        vc = comp.run(circuit, budget=budget)
        (
            result.num_cnots,
            result.depth,
            result.num_deps,
            result.esp,
        ) = _virtual_circuit_stats(vc)

        remove_mesurements(circuit)
        cuts = cut_circuit_wires(
            circuit=circuit,
            method="automatic",
            max_subcircuit_width=13,
            max_cuts=budget,
            num_subcircuits=[2],
        )
        (
            result.num_cnots_base,
            result.depth_base,
            result.num_deps_base,
            result.esp_base,
        ) = _cutqc_stats(cuts["subcircuits"], backend)

        result.num_cuts_base = cuts["num_cuts"]
        result.num_vgates = len(vc.virtual_gates)

        if run_on_hardware:
            # TODO cutqc run
            res, _ = qvm.run(vc, optimization_level=3, num_processes=8)
            fid = _compute_fidelity(res, vqe(circuit))
            result.fid = fid

        result.append_dict_to_csv(result_file)


def remove_mesurements(circuit: QuantumCircuit) -> None:
    circuit.remove_final_measurements(inplace=True)
    cregs = circuit.cregs.copy()
    for creg in cregs:
        circuit.remove_register(creg)


def prob_array_to_quasi_dist(prob_array: np.array) -> QuasiDistr:
    dist = QuasiDistr({})
    for i, prob in enumerate(prob_array):
        dist[i] = prob
    return dist


def _cutqc_stats(frags: list[QuantumCircuit], backend: BackendV2) -> tuple:
    frags = [transpile(frag, backend, optimization_level=3) for frag in frags]
    return (
        max(sum(1 for instr in frag if instr.operation.name == "cx") for frag in frags),
        max(frag.depth() for frag in frags),
        max(DAG(frag).num_dependencies() for frag in frags),
        min(_esp(frag) for frag in frags),
    )


def run_bench(benchname: str):
    circuits = get_circuits(benchname, (8, 25))

    provider = IBMProvider(instance="ibm-q-unibw/reservations/reservations")

    backend = provider.get_backend("ibmq_kolkata")

    try:
        run_cutqc_benchmark(
            f"bench/results/cutqc/{benchname}.csv",
            circuits,
            backend,
            size_to_reach=13,
            budget=4,
            run_on_hardware=True,
        )
    except Exception as e:
        print(e)


from multiprocessing.pool import ThreadPool

if __name__ == "__main__":
    for bname in [
        "qsvm",
        "wstate",
        "vqe_1",
        "vqe_2",
        "qaoa_b",
        "qaoa_r2",
        "hamsim_2",
        "hamsim_1",
        "twolocal_1",
    ]:
        run_bench(bname)
