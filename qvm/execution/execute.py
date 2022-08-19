from typing import Any, Dict, Optional

from qiskit import transpile
from qiskit.providers import Backend

from qvm.circuit import VirtualCircuit
from qvm.result import Result
from .knit import knit


def execute_virtual_circuit(
    virtual_circuit: VirtualCircuit,
    backend: Backend,
    transpile_kwargs: Optional[Dict[str, Any]] = None,
    run_kwargs: Optional[Dict[str, Any]] = None,
) -> Result:
    if not transpile_kwargs:
        transpile_kwargs = {}
    if not run_kwargs:
        run_kwargs = {}

    configured_circs = virtual_circuit.configured_circuits().items()
    conf_ids = [cf[0] for cf in configured_circs]
    circuits = [cf[1] for cf in configured_circs]

    # transpile and run
    t_circs = transpile(circuits, backend=backend, **transpile_kwargs)
    results = backend.run(t_circs, **run_kwargs).result().get_counts()

    if len(circuits) == 1:
        return Result.from_counts(results)
    return knit(
        {conf_id: Result.from_counts(res) for conf_id, res in zip(conf_ids, results)},
        virtual_circuit.virtual_gates,
    )


def execute_paralell_virtual_circuit(
    virtual_circuit: VirtualCircuit, backend: Backend
) -> Result:
    pass
