from typing import Any
from time import perf_counter
from dataclasses import dataclass
from multiprocessing import Pool

from qiskit.circuit import QuantumCircuit

from qvm.compiler.dag import DAG
from qvm.virtualizer import (
    Virtualizer,
    OneFragmentGateVirtualizer,
    TwoFragmentGateVirtualizer,
)
from qvm.util import insert_placeholders
from qvm.runners import QVMCircuitRunner


@dataclass
class RunTimer:
    exec_time: float
    knit_time: float


def _run_virtualizer(
    virtualizer: Virtualizer,
    runner: QVMCircuitRunner,
) -> tuple[dict[str, float], RunTimer]:
    frag_circs = virtualizer.fragments()

    frags, gen_circuits = zip(*frag_circs.items())

    all_circs = []
    for frag, args in virtualizer.instantiate():
        all_circs.append(
            [insert_placeholders(circ, args) for circ in gen_circuits[frag]]
        )

    with Pool() as pool:
        start = perf_counter()
        all_results = pool.map(runner.run, all_circs)
        exec_time = perf_counter() - start
        results = dict(zip(frags, all_results))
        start = perf_counter()
        res_distr = virtualizer.knit(results, pool=pool)
        knit_time = perf_counter() - start

    return res_distr.nearest_prob_distr(), RunTimer(exec_time, knit_time)


def run_vgate_circuit_as_one(
    virt_circuit: QuantumCircuit,
    runner: QVMCircuitRunner,
) -> tuple[dict[str, float], RunTimer]:
    dag = DAG(virt_circuit)
    dag.compact()
    virt_circuit = dag.to_circuit()
    virtualizer = OneFragmentGateVirtualizer(virt_circuit)
    return _run_virtualizer(virtualizer, runner)


def run_vgate_circuit(
    virt_circuit: QuantumCircuit,
    sampler: Sampler,
) -> tuple[dict[str, float], RunTimer]:
    dag = DAG(virt_circuit)
    dag.compact()
    dag.fragment()

    virt_circuit = dag.to_circuit()
    virtualizer: Virtualizer
    if len(virt_circuit.qregs) == 1:
        virtualizer = OneFragmentGateVirtualizer(virt_circuit)
        return _run_virtualizer(virtualizer, sampler)
    elif len(virt_circuit.qregs) == 2:
        virtualizer = TwoFragmentGateVirtualizer(virt_circuit)
        return _run_virtualizer(virtualizer, sampler)
    else:
        raise NotImplementedError(
            "Virtualization of circuits with more than 2 qubits is not yet supported"
        )
