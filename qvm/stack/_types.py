import abc
from dataclasses import dataclass

from qiskit.circuit import Barrier, Clbit, Parameter, QuantumCircuit
from qiskit.transpiler import CouplingMap

from qvm.quasi_distr import QuasiDistr


class PlaceholderGate(Barrier):
    def __init__(self, key: str, clbit: Clbit | None = None):
        super().__init__(num_qubits=1, label=key)
        self.key = key
        self.clbit = clbit


class QernelArgument:
    def __init__(
        self,
        insertions: dict[str, QuantumCircuit] | None = None,
        params: dict[Parameter, float] | None = None,
    ) -> None:
        self.insertions = insertions or {}
        self.params = params or {}
        if not all(
            circuit.num_qubits == 1 and circuit.num_clbits <= 1
            for circuit in self.insertions.values()
        ):
            raise ValueError("All circuits must have only one qubit")


def insert_placeholders(
    qernel: QuantumCircuit,
    input_: QernelArgument,
) -> QuantumCircuit:
    param_circuit = qernel.bind_parameters(input_.params)
    new_circuit = QuantumCircuit(
        *param_circuit.qregs,
        *param_circuit.cregs,
        name=param_circuit.name,
        global_phase=param_circuit.global_phase,
        metadata=param_circuit.metadata,
    )
    for cinstr in param_circuit.data:
        op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
        if isinstance(op, PlaceholderGate):
            if op.key not in input_.insertions:
                raise ValueError(f"Missing insertion for placeholder {op.key}")
            if op.clbit is not None:
                new_circuit.compose(
                    input_.insertions[op.key], qubits, clbits=[op.clbit], inplace=True
                )
            else:
                new_circuit.compose(input_.insertions[op.key], qubits, inplace=True)
        else:
            new_circuit.append(op, qubits, clbits)
    return new_circuit


@dataclass
class QVMJobMetadata:
    shots: int = 20000
    qpu_name: str | None = None
    initial_layout: list[int] | None = None
    vgates_to_spend: int = 0


class QPU(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        """Runs a qernel with the given arguments.

        Args:
            qernel (QuantumCircuit): The qernel to run.
            args (list[QernelArgument]): The arguments to run the qernel with.
            metadata (QVMJobMetadata): The metadata for the job.

        Returns:
            str: The job id.
        """
        ...

    @abc.abstractmethod
    def get_results(self, job_id: str) -> list[QuasiDistr]:
        """Returns the results of a job.

        Args:
            job_id (str): The job id.

        Returns:
            list[QuasiDistr]: The quasi-distribution results.
        """
        ...

    @abc.abstractmethod
    def coupling_map(self) -> CouplingMap:
        """Returns the coupling map of the QPU."""
        ...

    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Returns the number of qubits of the QPU."""
        ...


class QVMLayer(abc.ABC):
    @abc.abstractmethod
    def qpus(self) -> dict[str, QPU]:
        """Returns the QPUs the layer is aware of.

        Returns:
            dict[str, BackendV1]: The QPUs by name.
        """
        ...

    @abc.abstractmethod
    def run(
        self,
        qernel: QuantumCircuit,
        args: list[QernelArgument],
        metadata: QVMJobMetadata,
    ) -> str:
        """Runs a qernel with the given arguments.

        Args:
            qernel (QuantumCircuit): The qernel to run.
            args (list[QernelArgument]): The arguments to run the qernel with.
            metadata (QVMJobMetadata): The metadata for the job.

        Returns:
            str: The job id.
        """
        ...

    @abc.abstractmethod
    def get_results(self, job_id: str) -> list[QuasiDistr]:
        """Returns the results of a job.

        Args:
            job_id (str): The job id.

        Returns:
            list[QuasiDistr]: The quasi-distribution results.
        """
        ...
