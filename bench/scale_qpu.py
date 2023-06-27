from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import QiskitRuntimeService

from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner
from qvm.compiler import cut

from _backends import get_backend
from _circuits import get_circuits
from _run_experiment import run_experiment


def _run_scale(
    csv_name: str,
    circuit: QuantumCircuit,
    runner: QVMBackendRunner,
    backend: BackendV2,
    max_vgates: int,
    qpu_size: int,
    base_backend: BackendV2 | None = None,
):
    pass


def bench_scale_qpu(config: dict) -> None:
    backend: BackendV2 = get_backend(config["backend"])

    # # service = QiskitRuntimeService()
    # runner = LocalBackendRunner()

    # for bench in config["experiments"]:
    #     circuits = get_circuits(bench["name"], bench["param"], nums_qubits=[6, 8, 10])
    #     for circ in circuits:
    #         _run_vqr(bench["result_file"], circ, runner, backend, bench["max_vgates"])


if __name__ == "__main__":
    pass
