from tqdm import tqdm
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import QiskitRuntimeService

from qvm.qvm_runner import IBMBackendRunner, LocalBackendRunner
from qvm.virtualizer import Virtualizer
from qvm.compiler.virtualization.reduce_swap import ReduceSWAPCompiler


from _backends import get_backend
from _circuits import get_circuits
from _util import load_config
from _run_experiment import run_experiment


def run_bench_vqr(config: dict) -> None:
    backend: BackendV2 = get_backend(config["backend"])

    # service = QiskitRuntimeService()
    # runner = IBMBackendRunner(service)
    runner = LocalBackendRunner()

    progress = tqdm(total=len(config["experiments"]))
    progress.set_description("Running experiments")

    for bench in config["experiments"]:
        circuits = get_circuits(
            bench["name"], bench["param"], nums_qubits=config["nums_qubits"]
        )
        circ_progress = tqdm(total=len(circuits))
        circ_progress.set_description("Running circuits")

        comp = ReduceSWAPCompiler(
            backend, max_virtual_gates=bench["max_vgates"], reverse_order=False, max_distance=2
        )

        for circ in circuits:
            cut_circ = comp.run(circ)
            virt = Virtualizer(cut_circ)
            run_experiment(bench["result_file"], circ, virt, runner, backend)
            circ_progress.update(1)
        circ_progress.close()
        progress.update(1)


if __name__ == "__main__":
    run_bench_vqr(load_config())
