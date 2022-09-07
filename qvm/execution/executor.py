from typing import Dict

from qiskit.providers import Backend
import ray

from qvm.circuit import VirtualCircuit
from .frag_executor import FragmentExecutor
from .knit import knit


def execute(
    vc: VirtualCircuit, default_backend: Backend, shots: int = 10000
) -> Dict[str, int]:
    frag_execs = [
        FragmentExecutor.remote(vc, fragment, default_backend)  # type: ignore
        for fragment in vc.qregs
    ]
    futures = [frag_exec.execute.remote(shots) for frag_exec in frag_execs]
    ray.get(futures)

    future = knit.remote(frag_execs, vc.virtual_gates)
    return ray.get(future).counts(shots)
