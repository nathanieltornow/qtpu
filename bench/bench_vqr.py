from dataclasses import dataclass, asdict
from tqdm import tqdm
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import QiskitRuntimeService

from qvm.qvm_runner import IBMBackendRunner, LocalBackendRunner
from qvm.virtualizer import Virtualizer
from qvm.compiler.virtualization.reduce_swap import ReduceSWAPCompiler


from _backends import get_backend
from _circuits import get_circuits
from _util import load_config, append_to_csv_file
from _run_experiment import run_experiment, get_circuit_properties
from qiskit_ibm_runtime import QiskitRuntimeService


def run_vqr_stats(config: dict) -> None:
    backend: BackendV2 = get_backend(config["backend"])

    progress = tqdm(total=len(config["experiments"]))
    progress.set_description("Running experiments")

    for bench in config["experiments"]:
        circuits = get_circuits(
            bench["name"], bench["param"], nums_qubits=config["nums_qubits"]
        )
        circ_progress = tqdm(total=len(circuits))
        circ_progress.set_description("Running circuits")

        comp = ReduceSWAPCompiler(
            backend,
            max_virtual_gates=bench["max_vgates"],
            reverse_order=True,
            max_distance=2,
        )

        for circ in circuits:
            cut_circ = comp.run(circ)
            virt = Virtualizer(cut_circ)
            bench_stat = get_circuit_properties(circ, virt, backend)
            append_to_csv_file(bench["result_file"], asdict(bench_stat))
            circ_progress.update(1)
        circ_progress.close()
        progress.update(1)


# def run_bench_vqr(config: dict) -> None:
#     backend: BackendV2 = get_backend(config["backend"])

#     service = QiskitRuntimeService(
#         channel="ibm_quantum",
#         instance="ibm-q-research-2/tu-munich-1/main",
#     )
#     runner = IBMBackendRunner(service)
#     # runner = LocalBackendRunner()

#     progress = tqdm(total=len(config["experiments"]))
#     progress.set_description("Running experiments")

#     for bench in config["experiments"]:
#         circuits = get_circuits(
#             bench["name"], bench["param"], nums_qubits=config["nums_qubits"]
#         )
#         circ_progress = tqdm(total=len(circuits))
#         circ_progress.set_description("Running circuits")

#         comp = ReduceSWAPCompiler(
#             backend,
#             max_virtual_gates=bench["max_vgates"],
#             reverse_order=True,
#             max_distance=2,
#         )

#         for circ in circuits:
#             cut_circ = comp.run(circ)
#             virt = Virtualizer(cut_circ)
#             run_experiment(bench["result_file"], circ, virt, runner, backend)
#             circ_progress.update(1)
#         circ_progress.close()
#         progress.update(1)


if __name__ == "__main__":
    run_vqr_stats(load_config())
