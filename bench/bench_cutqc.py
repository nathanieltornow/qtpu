import os
import csv
from dataclasses import dataclass, asdict

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.compiler import transpile

import qvm
from qvm import QVMCompiler, QuasiDistr
from qvm.compiler.virtualization import OptimalDecompositionPass
from qvm.compiler.distr_transpiler.backend_mapper import BasicBackendMapper
from qvm.compiler.dag import DAG
from qiskit.providers.fake_provider import FakeKolkataV2

from cutqc import (
    cut_circuit_wires,
    evaluate_subcircuits,
    reconstruct_full_distribution,
)

from bench_main import (
    calculate_fidelity,
    _esp,
    _virtual_circuit_stats,
)
from cutqc.wire_cutting_evaluation import set_backend_config
from circuits.circuits import get_circuits


@dataclass
class BenchmarkResult:
    num_qubits: int
    fid: float = np.nan
    fid_cutqc: float = np.nan
    fid_base: float = np.nan
    esp: float = np.nan
    esp_cutqc: float = np.nan
    esp_base: float = np.nan
    num_cnots: int = np.nan
    num_cnots_cutqc: int = np.nan
    num_cnots_base: int = np.nan
    depth: int = np.nan
    depth_cutqc: int = np.nan
    depth_base: int = np.nan
    num_deps: int = np.nan
    num_deps_cutqc: int = np.nan
    num_deps_base: int = np.nan
    num_vgates: int = np.nan
    num_cuts_cutqc: int = np.nan

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
    set_backend_config(backend, 3)

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
            max_subcircuit_width=7,
            max_cuts=3,
            num_subcircuits=[2],
            verbose=False,
        )
        (
            result.num_cnots_cutqc,
            result.depth_cutqc,
            result.num_deps_cutqc,
            result.esp_cutqc,
        ) = _cutqc_stats(cuts["subcircuits"], backend)

        result.num_cuts_cutqc = cuts["num_cuts"]
        result.num_vgates = len(vc.virtual_gates)

        circ_cp = circuit.copy()
        circ_cp.measure_all()

        t_circ = transpile(circ_cp, backend, optimization_level=3)
        result.num_cnots_base = sum(
            1 for instr in t_circ if instr.operation.name == "cx"
        )
        result.depth_base = t_circ.depth()
        result.num_deps_base = DAG(circ_cp).num_dependencies()
        result.esp_base = _esp(t_circ)

        if run_on_hardware:
            probs = evaluate_subcircuits(cuts)
            cutqc_res = prob_array_to_quasi_dist(
                reconstruct_full_distribution(circuit, probs, cuts, num_threads=8)
            ).nearest_probability_distribution()

            circuit.measure_all()

            qvm_res, _ = qvm.run(vc, optimization_level=3, num_processes=8)

            fid = calculate_fidelity(circuit, qvm_res)
            fid_cutqc = calculate_fidelity(circuit, cutqc_res)
            result.fid = fid
            result.fid_cutqc = fid_cutqc

            base_res = backend.run(t_circ, shots=20000).result().get_counts()
            base_probs = QuasiDistr.from_counts(
                base_res
            ).nearest_probability_distribution()
            fid_base = calculate_fidelity(circuit, base_probs)
            result.fid_base = fid_base

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
    circuits = get_circuits(benchname, (8, 13))

    backend = FakeKolkataV2()

    run_cutqc_benchmark(
        f"bench/results/cutqc_fake/{benchname}.csv",
        circuits,
        backend,
        size_to_reach=7,
        budget=2,
        run_on_hardware=False,
    )


if __name__ == "__main__":
    for bname in [
        "hamsim_1",
        "vqe_1",
        "vqe_2",
        "hamsim_2",
        "wstate",
        "twolocal_1",
        "qaoa_r2",
        "qsvm",
    ]:
        print(
            f"-----------------------------------\n{bname}\n-----------------------------------"
        )
        run_bench(bname)
