from qiskit.circuit import QuantumCircuit

from qvm.core import insert_placeholders, QuasiDistr, Argument
from .types import QVMInterface, QPU


class SingleQPUWrapper(QPU):
    def __init__(self, qpu: QPU) -> None:
        self._qpu = qpu

    def _run(
        self,
        circuit: QuantumCircuit,
        args: list[Argument],
        shots: int = 20000,
        max_overhead: int = 300,
    ) -> str:
        if len(args) == 1:
            circuit = insert_placeholders(circuit, args[0])
            if circuit.num_qubits > self._qpu.num_qubits():
                pass
            pass
        else:
            raise NotImplementedError()

    def results(self, job_id: str) -> list[QuasiDistr]:
        return super().results()
