from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options

from qiskit.providers import BackendV2
from qiskit.providers.fake_provider import FakeKolkataV2

from qvm.runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner

from _backends import get_simulator_session, generate_default_simulator_options
from _util import load_config


def _run_vqr(
    csv_name: str,
    circuit: QuantumCircuit,
    backend_runner: QVMBackendRunner,
    backend: BackendV2,
) -> None:
    pass


def run_bench_vqr(config: dict) -> None:
    service = QiskitRuntimeService()
    session = get_simulator_session(service)

    sampler = Sampler(session=session, options=generate_default_simulator_options())


if __name__ == "__main__":
    run_bench_vqr(load_config())
