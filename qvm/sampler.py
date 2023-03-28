import numpy as np

from qiskit import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Sampler, SamplerResult, BackendSampler
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.result.distributions.quasi import QuasiDistribution

from qvm.stack._types import QVMLayer, QVMJobMetadata


class QVMSampler(Sampler):
    def __init__(self, qvm_stack: QVMLayer):
        super().__init__(None, None, None)
        self._qvm_stack = qvm_stack

    def _call(
        self,
        circuits: list[QuantumCircuit],
        shots: int | None = None,
    ) -> SamplerResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        shots = shots or 20000

        # Initialize metadata
        metadata = [{"shots": shots} for _ in range(len(circuits))]

        jobs = []
        for circ in circuits:
            jobs.append(self._qvm_stack.run(circ, [], QVMJobMetadata(shots=shots)))

        print(jobs)
        
        results = [self._qvm_stack.get_results(job)[0] for job in jobs]

        quasis = [QuasiDistribution(result) for result in results]

        return SamplerResult(quasis, metadata)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> PrimitiveJob:
        circuits = [
            circuit.bind_parameters(dict(zip(circuit.parameters, values)))
            for circuit, values in zip(circuits, parameter_values)
        ]

        job = PrimitiveJob(self._call, circuits, run_options.pop("shots", None))
        job.submit()
        return job

    def __new__(  # pylint: disable=signature-differs
        cls,
        qvm_stack: QVMLayer,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self = super().__new__(cls)
        return self
