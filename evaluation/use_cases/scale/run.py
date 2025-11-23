from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from mqt.bench import get_benchmark_indep

import benchkit as bk
from evaluation.use_cases.scale.qac import cut_circuit, run_qac
from evaluation.use_cases.scale.qtpu import run_qtpu

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


@bk.foreach(circuit_size=[10, 20, 30, 40])
@bk.foreach(subcirc_size=[10])
@bk.foreach(bench=["qnn", "wstate"])
@bk.foreach(num_samples=[100000])
@bk.foreach(_repeat=list(range(5)))
@bk.log("logs/01_scale_qac.jsonl")
def scale_qac_bench(
    bench: str, circuit_size: int, subcirc_size: int, num_samples: int, _repeat: int
) -> QuantumCircuit:
    circuit = get_benchmark_indep(bench, circuit_size=circuit_size, opt_level=3)

    start = perf_counter()
    cut_circ, observables = cut_circuit(circuit, max_qubits=subcirc_size)
    cut_time = perf_counter() - start

    qac_metrics = run_qac(
        cut_circ,
        observables,
        num_samples=num_samples,
        shots=1000,
    )

    return {
        "qac.cut_time": cut_time,
        **qac_metrics,
    }


@bk.foreach(circuit_size=[10, 20, 30, 40, 50, 60, 70, 80])
@bk.foreach(subcirc_size=[10])
@bk.foreach(bench=["qnn", "wstate"])
@bk.foreach(_repeat=list(range(5)))
@bk.log("logs/01_scale_qtpu.jsonl")
def scale_qtpu_bench(
    bench: str, circuit_size: int, subcirc_size: int, _repeat: int
) -> QuantumCircuit:
    circuit = get_benchmark_indep(bench, circuit_size=circuit_size, opt_level=3)

    cut_circ, _ = cut_circuit(circuit, max_qubits=subcirc_size)

    qtpu_metrics = run_qtpu(cut_circ)

    return {
        **qtpu_metrics,
    }


if __name__ == "__main__":
    scale_qac_bench()
    scale_qtpu_bench()