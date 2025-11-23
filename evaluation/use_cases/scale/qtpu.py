from __future__ import annotations

import tracemalloc
from time import perf_counter
from typing import TYPE_CHECKING

from qiskit_ibm_runtime.fake_provider import FakeMontrealV2

import qtpu
from evaluation.analysis import estimate_runtime

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def run_qtpu(circuit: QuantumCircuit) -> dict[str, float]:
    tracemalloc.start()
    start = perf_counter()
    htn = qtpu.circuit_to_hybrid_tn(circuit)
    qtpu_gen_time = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    all_circuits = []
    for qt in htn.qtensors:
        all_circuits += qt.flat()

    quantum_time = estimate_runtime(
        circuits=all_circuits,
        backend=FakeMontrealV2(),
        shots=1000,
    )

    tn = htn.to_dummy_tn()
    start = perf_counter()
    tn.contract(all, optimize="auto-hq", output_inds=[])
    classical_time = perf_counter() - start

    return {
        "qtpu.generation_time": qtpu_gen_time,
        "qtpu.generation_memory": peak,
        "qtpu.quantum_time": quantum_time,
        "qtpu.classical_time": classical_time,
        "qtpu.num_subcircuits": len(htn.subcircuits),
        "qtpu.num_experiments": len(all_circuits),
    }
