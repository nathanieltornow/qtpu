from typing import Any
from time import perf_counter
from dataclasses import dataclass
from multiprocessing import Pool

from qiskit.primitives import Sampler
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile

from qvm.dag import DAG
from qvm.virtualizer import (
    Virtualizer,
    OneFragmentGateVirtualizer,
    TwoFragmentGateVirtualizer,
)
from qvm.quasi_distr import QuasiDistr
from qvm.compiler.util import fragment_dag
from qvm.util import insert_placeholders


@dataclass
class RunTimer:
    exec_time: float
    knit_time: float


def _run_virtualizer(
    virtualizer: Virtualizer,
    sampler: Sampler,
    transpile_args: dict[str, Any],
    run_args: dict[str, Any],
) -> tuple[dict[str, float], RunTimer]:
    frag_circs = virtualizer.fragments()
    print(len(frag_circs))
    start = perf_counter()
    jobs = {}
    for fragment, args in virtualizer.instantiate().items():
        t_circ = transpile(frag_circs[fragment], **transpile_args)
        circuits = [insert_placeholders(frag_circs[fragment], arg) for arg in args]
        jobs[fragment] = sampler.run(circuits, **run_args)

    results = {}
    for fragment, job in jobs.items():
        dists = job.result().quasi_dists
        dists = [dists] if isinstance(dists, dict) else dists
        results[fragment] = [
            QuasiDistr.from_sampler_distr(
                count, sum(len(creg) for creg in frag_circs[fragment].cregs)
            )
            for count in dists
        ]
    exec_time = perf_counter() - start
    start = perf_counter()
    with Pool() as pool:
        res_distr = virtualizer.knit(results, pool=pool)
    knit_time = perf_counter() - start
    return res_distr.nearest_prob_distr(), RunTimer(exec_time, knit_time)


def run_vgate_circuit_as_one(
    virt_circuit: QuantumCircuit,
    sampler: Sampler,
    transpile_args: dict[str, Any] | None = None,
    run_args: dict[str, Any] | None = None,
) -> tuple[dict[str, float], RunTimer]:
    if transpile_args is None:
        transpile_args = {}
    if run_args is None:
        run_args = {}
    dag = DAG(virt_circuit)
    dag.compact()
    virt_circuit = dag.to_circuit()
    virtualizer = OneFragmentGateVirtualizer(virt_circuit)
    return _run_virtualizer(virtualizer, sampler, transpile_args, run_args)


def run_vgate_circuit(
    virt_circuit: QuantumCircuit,
    sampler: Sampler,
    transpile_args: dict[str, Any] | None = None,
    run_args: dict[str, Any] | None = None,
) -> tuple[dict[str, float], RunTimer]:
    if transpile_args is None:
        transpile_args = {}
    if run_args is None:
        run_args = {}

    dag = DAG(virt_circuit)
    dag.compact()
    fragment_dag(dag)

    virt_circuit = dag.to_circuit()
    virtualizer: Virtualizer
    if len(virt_circuit.qregs) == 1:
        virtualizer = OneFragmentGateVirtualizer(virt_circuit)
        return _run_virtualizer(virtualizer, sampler, transpile_args, run_args)
    elif len(virt_circuit.qregs) == 2:
        virtualizer = TwoFragmentGateVirtualizer(virt_circuit)
        return _run_virtualizer(virtualizer, sampler, transpile_args, run_args)
    else:
        raise NotImplementedError(
            "Virtualization of circuits with more than 2 qubits is not yet supported"
        )
