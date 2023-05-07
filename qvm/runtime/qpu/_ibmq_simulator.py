from qiskit.providers.ibmq import AccountProvider
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap

from qvm.core import insert_placeholders, QuasiDistr, Argument
from qvm.runtime.types import QPU


class IBMQSimulator(QPU):
    def __init__(self, provider: AccountProvider):
        self._backend = provider.get_backend("ibmq_qasm_simulator")

    def coupling_map(self) -> CouplingMap:
        return CouplingMap.from_full(self.num_qubits())

    def num_qubits(self) -> int:
        return 32

    def _run(
        self,
        circuit: QuantumCircuit,
        args: list[Argument],
        shots: int = 20000,
        max_overhead: int = 300,
    ) -> str:
        t_circ = transpile(circuit, backend=self._backend, optimization_level=0)
        circuits_to_run = [insert_placeholders(t_circ, arg) for arg in args]
        manager = IBMQJobManager()
        