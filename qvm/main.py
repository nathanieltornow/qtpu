import logging
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
# from ray.util.multiprocessing import Pool

from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qvm.quasi_distr import QuasiDistr
from .runtime.virtualizer import Virtualizer


def _run_circuits(circuits: list[QuantumCircuit], shots: int) -> list[QuasiDistr]:
    counts = AerSimulator().run(circuits, shots=shots).result().get_counts()
    counts = [counts] if isinstance(counts, dict) else counts
    return [QuasiDistr.from_counts(counts=count, shots=shots) for count in counts]


def run_on_sim(circuit: QuantumCircuit, shots: int = 10000) -> QuasiDistr:
    virtualizer = Virtualizer(circuit)

    circs = {}
    for qreg in circuit.qregs:
        circs[qreg] = virtualizer.instantiations(qreg)

    qregs, circ_lists = zip(*circs.items())
    with Pool(cpu_count()) as p:
        logging.info("Running circuits on %d processes", cpu_count())
        all_results = p.starmap(
            _run_circuits, [(circuits, shots) for circuits in circ_lists]
        )
        for qreg, results in zip(qregs, all_results):
            virtualizer.put_results(qreg, results)
        logging.info("Knitting results")
        final_result = virtualizer.knit(p)
    return final_result
