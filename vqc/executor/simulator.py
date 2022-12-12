from functools import partial
from typing import Dict, List, Optional

from qiskit import Aer
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers.models import BackendConfiguration
from qiskit_aer import AerSimulator

from vqc.prob_distr import Counts
from vqc.types import Executor


class Simulator(Executor):
    def __init__(
        self,
        backend_configuration: Optional[BackendConfiguration] = None,
        shots: int = 10000,
    ) -> None:
        self._shots = shots
        self._simulator_instance = (
            Aer.get_backend("statevector_simulator")
            if backend_configuration is None
            else AerSimulator.from_backend(backend_configuration)
        )

        # Use `partial` to freeze the shots parameters
        self._simulator_run_fn = partial(
            self._simulator_instance.run, shots=self._shots
        )

    def execute(
        self,
        sampled_circuits: Dict[str, List[QuantumCircuit]],
        parameter_binding_dict: Dict[Parameter, float] = {},
    ) -> Dict[str, List[Counts]]:
        results = {}
        for name, frag_circs in sampled_circuits.items():
            # Filter out the non-existent parameters for a specific fragment
            filtered_keys = parameter_binding_dict.keys() & frag_circs[0].parameters

            filtered_parameters = {
                key: parameter_binding_dict[key] for key in filtered_keys
            }

            # Run the simulation
            results[name] = (
                self._simulator_run_fn(
                    [
                        circ.bind_parameters(filtered_parameters).decompose()
                        for circ in frag_circs
                    ],
                )
                .result()
                .get_counts()
            )
        return results
