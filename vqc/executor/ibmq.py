import itertools

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.ibmq import AccountProvider
from qiskit.providers.ibmq.managed import IBMQJobManager

from vqc.prob_distr import Counts

from .executor import Executor


class IBMQExecutor(Executor):
    def __init__(
        self, provider: AccountProvider, backend_name: str, optimization_level: int = 2
    ):
        self._provider = provider
        self._backend = provider.get_backend(backend_name)
        self._optimization_level = optimization_level

    def execute(
        self, sampled_circuits: dict[str, list[QuantumCircuit]]
    ) -> dict[str, list[Counts]]:
        # Put all circuits in a single list for a single job
        frag_names, circuits = zip(*sampled_circuits.items())
        circuits_per_frag = [len(circuits) for circuits in circuits]
        all_circuits = list(itertools.chain(*circuits))
        all_circuits = transpile(
            all_circuits, self._backend, optimization_level=self._optimization_level
        )
        job_manager = IBMQJobManager()
        counts = (
            job_manager.run(all_circuits, backend=self._backend).results().get_counts()
        )
        results = {}
        for i, num in enumerate(circuits_per_frag):
            results[frag_names[i]] = counts[:num]
            counts = counts[num:]
        return results
