from dataclasses import dataclass
from time import perf_counter

from qiskit.compiler import transpile

from qvm.quasi_distr import QuasiDistr
from qvm.runtime.virtualizer import Virtualizer
from qvm.virtual_circuit import VirtualCircuit


@dataclass
class RuntimeInfo:
    qpu_time: float
    knit_time: float


def run(
    virtual_circuit: VirtualCircuit, shots: int = 20000
) -> tuple[dict[int, float], RuntimeInfo]:
    virt = Virtualizer(virtual_circuit)
    instantiations = virt.instantiations()

    meta = virtual_circuit.metadata

    now = perf_counter()

    jobs = {}
    for frag, insts in instantiations.items():
        insts = [
            transpile(inst, backend=meta[frag].backend, optimization_level=0)
            for inst in insts
        ]
        job = meta[frag].backend.run(insts, shots=shots)
        jobs[frag] = job

    results = {}
    for frag, job in jobs.items():
        counts = job.result().get_counts()
        counts = counts if isinstance(counts, list) else [counts]
        dists = [QuasiDistr.from_counts(c) for c in counts]
        results[frag] = dists

    runtime = perf_counter() - now
    res = virt.knit(results)
    knit_time = perf_counter() - now - runtime
    return res, RuntimeInfo(runtime, knit_time)
