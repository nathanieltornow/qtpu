from dataclasses import dataclass
from time import perf_counter

import numpy as np
from qiskit.compiler import transpile

from qvm.quasi_distr import QuasiDistr

from qvm.runtime.virtualizer import Virtualizer
from qvm.virtual_circuit import VirtualCircuit


@dataclass
class RuntimeInfo:
    qpu_time: float
    knit_time: float


def run(
    virtual_circuit: VirtualCircuit,
    shots: int = 20000,
    optimization_level: int = 0,
) -> tuple[dict[int, float], RuntimeInfo]:
    """Run a virtual circuit.

    Args:
        virtual_circuit (VirtualCircuit): The virtual circuit to run.
        shots (int, optional):
            The number of shots for each fragment instantiation. Defaults to 20000.
        optimization_level (int, optional):
            The pre-run optimization level. Since optimization should idealy be
            done on the respective fragments, it defaults to 0. Defaults to 0.

    Returns:
        tuple[dict[int, float], RuntimeInfo]:
            The resulting distribution and a runtime info for benchmarking.
    """

    virt = Virtualizer(virtual_circuit)

    meta = virtual_circuit.metadata

    print(f"Running {virtual_circuit.num_instantiations} instantiations...")

    now = perf_counter()

    jobs = {}
    for frag, insts in virt.instantiations().items():
        insts = [
            transpile(
                inst, backend=meta[frag].backend, optimization_level=optimization_level
            )
            for inst in insts
        ]
        job = meta[frag].backend.run(insts, shots=shots)
        jobs[frag] = job

    results = {}
    for frag, job in jobs.items():
        counts = job.result().get_counts()
        counts = counts if isinstance(counts, list) else [counts]
        res = np.array([z_expval(count) for count in counts], dtype=np.float64)
        results[frag] = res

    runtime = perf_counter() - now

    print("Knitting results...")
    res = virt.knit(results)
    knit_time = perf_counter() - now - runtime
    return res, RuntimeInfo(runtime, knit_time)


def z_expval(counts: dict[str, int]) -> float:
    expval = 0.0
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        parity = 1 - 2 * int(bitstring.count("1") % 2)
        expval += parity * (count / shots)
    return expval
