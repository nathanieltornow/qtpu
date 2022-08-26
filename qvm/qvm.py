from typing import Any, Dict
from qiskit.providers import Backend

from qvm.circuit import VirtualCircuit
from qvm.execution.exec import execute_fragmented_circuit, execute_virtual_circuits


def execute_virtual_circuit(
    virtual_circuit: VirtualCircuit,
    backend: Backend,
    transpile_flags: Dict[str, Any],
    exec_flags: Dict[str, Any],
) -> Dict[str, int]:
    shots = 8192
    if "shots" in exec_flags:
        shots = exec_flags["shots"]
    else:
        exec_flags["shots"] = shots

    res = execute_virtual_circuits(
        [virtual_circuit], backend, transpile_flags, exec_flags
    )[0]
    return res.counts(shots)
