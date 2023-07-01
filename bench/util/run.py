import os
import csv
from dataclasses import dataclass, asdict

from tqdm import tqdm
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.compiler import transpile

from qvm.qvm_runner import QVMBackendRunner
from qvm.run import run_virtualizer
from qvm.virtual_circuit import VirtualCircuit
from qvm.compiler import CutCompiler


from ._util import compute_fidelity, get_num_cnots


@dataclass
class Benchmark:
    circuits: list[QuantumCircuit]
    backend: BackendV2
    result_file: str
    virt_compiler: CutCompiler
    base_backend: BackendV2 | None = None


@dataclass
class BenchmarkResult:
    num_qubits: int
    h_fid: float = 0.0
    h_fid_base: float = 0.0
    tv_fid: float = 0.0
    tv_fid_base: float = 0.0
    num_cnots: int = 0
    num_cnots_base: int = 0
    depth: int = 0
    depth_base: int = 0
    num_vgates: int = 0
    num_fragments: int = 0
    run_time: float = 0.0
    knit_time: float = 0.0

    def append_to_csv(self, filepath: str) -> None:
        data = asdict(self)
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as csv_file:
                csv.DictWriter(csv_file, fieldnames=data.keys()).writeheader()
                csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)
            return

        with open(filepath, "a") as csv_file:
            csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)


def run_benchmark(bench: Benchmark, runner: QVMBackendRunner | None = None) -> None:
    progress = tqdm(total=len(bench.circuits))
    progress.set_description("Running Bench Circuits")

    for circ in bench.circuits:
        cut_circuit = bench.virt_compiler.run(circ)
        virt = VirtualCircuit(cut_circuit)
        res = _run_experiment(
            circ,
            virt,
            bench.backend,
            runner=runner,
            base_backend=bench.base_backend,
        )
        res.append_to_csv(bench.result_file)
        progress.update(1)


def _run_experiment(
    original_circuit: QuantumCircuit,
    virt: VirtualCircuit,
    backend: BackendV2,
    runner: QVMBackendRunner | None = None,
    base_backend: BackendV2 | None = None,
) -> BenchmarkResult:
    if base_backend is None:
        base_backend = backend

    t_circ = transpile(original_circuit, backend=base_backend, optimization_level=3)

    num_cnots, depth = _virtualizer_stats(virt, backend)
    num_cnots_base, depth_base = get_num_cnots(t_circ), t_circ.depth()
    num_vgates = len(virt._vgate_instrs)
    if num_vgates > 4:
        print(f"WARN: {num_vgates}")

    num_fragments = len(virt.fragment_circuits)

    if runner is None or num_vgates > 4:
        return BenchmarkResult(
            num_qubits=original_circuit.num_qubits,
            num_cnots=num_cnots,
            num_cnots_base=num_cnots_base,
            depth=depth,
            depth_base=depth_base,
            num_vgates=num_vgates,
            num_fragments=num_fragments,
        )

    result, timing = run_virtualizer(virt, runner, backend)

    h_fid, tv_fid = 0.0, 0.0
    if original_circuit.num_qubits < 25:
        h_fid, tv_fid = compute_fidelity(original_circuit, result, runner)

    h_fid_base, tv_fid_base = 0.0, 0.0

    if original_circuit.num_qubits < 25:
        job_id = runner.run([t_circ], base_backend)
        noisy_base_res = runner.get_results(job_id)[
            0
        ].nearest_probability_distribution()
        h_fid_base, tv_fid_base = compute_fidelity(
            original_circuit, noisy_base_res, runner
        )

    return BenchmarkResult(
        num_qubits=original_circuit.num_qubits,
        num_cnots=num_cnots,
        num_cnots_base=num_cnots_base,
        depth=depth,
        depth_base=depth_base,
        num_vgates=num_vgates,
        h_fid=h_fid,
        h_fid_base=h_fid_base,
        tv_fid=tv_fid,
        tv_fid_base=tv_fid_base,
        run_time=timing.run_time,
        knit_time=timing.knit_time,
        num_fragments=num_fragments,
    )


def _virtualizer_stats(virtualizer: VirtualCircuit, backend: BackendV2) -> tuple[int, int]:
    frag_circs = list(virtualizer.fragment_circuits.values())
    frag_circs = [transpile(circ, backend, optimization_level=3) for circ in frag_circs]
    num_cnots = max(get_num_cnots(circ) for circ in frag_circs)
    depth = max(circ.depth() for circ in frag_circs)
    return num_cnots, depth
