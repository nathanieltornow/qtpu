from time import perf_counter
import networkx as nx
from qiskit.providers import BackendV2
from qiskit.circuit import QuantumCircuit

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler.virtualization.gate_decomp import OptimalGateDecomposer
from qvm.virtual_circuit import VirtualCircuit
from qvm.run import run_virtualizer

from util.circuits import hamsim, qaoa
from util.run import BenchmarkResult
from util._util import enable_logging, append_dict_to_csv


def _run_circuit(
    circuit: QuantumCircuit,
    runner: QVMBackendRunner,
    fragment_size: int,
    run_base: bool = True,
) -> BenchmarkResult:
    comp = OptimalGateDecomposer(fragment_size)
    cut_circ = comp.run(circuit)
    virt = VirtualCircuit(cut_circ)
    _, run_time_info = run_virtualizer(virt, runner)
    run_time_base = 0.0
    if run_base:
        now = perf_counter()
        job_id = runner.run(circuit)
        runner.get_results(job_id)
        run_time_base = perf_counter() - now
    return BenchmarkResult(
        num_qubits=circuit.num_qubits,
        run_time=run_time_info.run_time,
        knit_time=run_time_info.knit_time,
        run_time_base=run_time_base,
    )


def bench_scale_sim(
    result_file: str,
    circuits: list[QuantumCircuit],
    runner: QVMBackendRunner,
    fragment_size: int,
    sim_limit: int = 30,
) -> None:
    for circ in circuits:
        bench_res = _run_circuit(
            circ, runner, fragment_size, circ.num_qubits <= sim_limit
        )
        bench_res.append_to_csv(result_file)


def main() -> None:
    result_dir = f"bench/results/scale_sim"

    runner = LocalBackendRunner()

    frag_size = 10

    circuits = [hamsim(i, 1) for i in range(frag_size, 51, 10)]
    bench_scale_sim(
        result_file=f"{result_dir}/hamsim_1_{frag_size}.csv",
        circuits=circuits,
        runner=runner,
        fragment_size=frag_size,
        sim_limit=20,
    )


if __name__ == "__main__":
    enable_logging()
    main()
