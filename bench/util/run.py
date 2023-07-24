from time import perf_counter
from dataclasses import dataclass, asdict

from tqdm import tqdm
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.compiler import transpile

from qvm.qvm_runner import QVMBackendRunner
from qvm.run import run_virtualizer
from qvm.virtual_circuit import VirtualCircuit
from qvm.compiler import QVMCompiler
from qvm.compiler.dag import DAG


from ._util import (
    compute_fidelity,
    get_num_cnots,
    get_circuit_depth,
    append_dict_to_csv,
)

DYNAMIC_CIRCUITS = True


@dataclass
class Benchmark:
    circuits: list[QuantumCircuit]
    backend: BackendV2
    result_file: str
    compiler: QVMCompiler
    base_compiler: QVMCompiler | None = None
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
    num_deps: int = 0
    num_deps_base: int = 0
    num_vgates: int = 0
    num_fragments: int = 0
    num_instances: int = 0
    run_time: float = 0.0
    knit_time: float = 0.0
    run_time_base: float = 0.0

    def append_to_csv(self, filepath: str) -> None:
        append_dict_to_csv(filepath, asdict(self))


def run_benchmark(bench: Benchmark, runner: QVMBackendRunner | None = None) -> None:
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


def _run_experiment(
    original_circuit: QuantumCircuit,
    virt: VirtualCircuit,
    base_virt: VirtualCircuit,
    backend: BackendV2,
    runner: QVMBackendRunner | None = None,
    base_backend: BackendV2 | None = None,
    run_base: bool = True,
) -> BenchmarkResult:
    if base_backend is None:
        base_backend = backend

    num_cnots, depth, num_deps = _virtualizer_stats(virt, backend)

    num_cnots_base, depth_base, num_deps_base = _virtualizer_stats(base_virt, backend)

    num_vgates = len(virt._vgate_instrs)
    num_fragments = len(virt.fragment_circuits)

    num_instances = 0
    for frag in virt.fragment_circuits.keys():
        num_instances += len(virt.get_instance_labels(frag))

    if runner is None or num_vgates > 4:
        # NOTE: when num_vgates > 4, the virtual circuit is too large to run on the QVM
        return BenchmarkResult(
            num_qubits=original_circuit.num_qubits,
            num_cnots=num_cnots,
            num_cnots_base=num_cnots_base,
            depth=depth,
            depth_base=depth_base,
            num_deps=num_deps,
            num_deps_base=num_deps_base,
            num_vgates=num_vgates,
            num_fragments=num_fragments,
            num_instances=num_instances,
        )

    result, timing = run_virtualizer(virt, runner, backend)

    h_fid, tv_fid = 0.0, 0.0
    if original_circuit.num_qubits < 25:
        h_fid, tv_fid = compute_fidelity(original_circuit, result, runner)

    h_fid_base, tv_fid_base = 0.0, 0.0

    base_result, base_time = None, 0.0
    if run_base and original_circuit.num_qubits < base_backend.num_qubits:
        base_result, base_timing = run_virtualizer(base_virt, runner, base_backend)
        base_time = base_timing.run_time

    if base_result is not None and run_base:
        h_fid_base, tv_fid_base = compute_fidelity(
            original_circuit, base_result, runner
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
        num_instances=num_instances,
        run_time_base=base_time,
    )


def _virtualizer_stats(
    virtualizer: VirtualCircuit, backend: BackendV2
) -> tuple[int, int, int]:
    frag_circs = list(virtualizer.fragment_circuits.values())
    num_deps = max(DAG(circ).num_dependencies() for circ in frag_circs)

    try:
        frag_circs = [
            transpile(circ, backend, optimization_level=3) for circ in frag_circs
        ]
    except Exception:
        print("Transpilation failed")
        return 0, 0, 0
    num_cnots = max(get_num_cnots(circ) for circ in frag_circs)
    depth = max(get_circuit_depth(circ) for circ in frag_circs)

    return num_cnots, depth, num_deps
