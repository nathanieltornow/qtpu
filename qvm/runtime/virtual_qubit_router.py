from qiskit.circuit import QuantumCircuit

from qvm.core import cut, QuasiDistr, Argument, insert_placeholders, fragment_circuit
from qvm.runtime.types import QVMInterface, QPU


class VirtualQubitRouter(QVMInterface):
    def __init__(self, qpu: QPU) -> None:
        self._qpu = qpu

    def _run(
        self,
        circuit: QuantumCircuit,
        args: list[Argument],
        shots: int = 20000,
        max_overhead: int = 300,
    ) -> str:
        return super()._run(circuit, args, shots, max_overhead)

    def results(self, job_id: str) -> list[QuasiDistr]:
        return super().results(job_id)
