import logging
import time
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qvm.quasi_distr import QuasiDistr

from .runtime.virtualizer import Virtualizer

# from ray.util.multiprocessing import Pool




def _run_circuits(circuits: list[QuantumCircuit], shots: int) -> list[QuasiDistr]:
    counts = AerSimulator().run(circuits, shots=shots).result().get_counts()
    counts = [counts] if isinstance(counts, dict) else counts
    return [QuasiDistr.from_counts(counts=count, shots=shots) for count in counts]


def run_on_sim(circuit: QuantumCircuit, shots: int = 10000) -> QuasiDistr:
    virtualizer = Virtualizer(circuit)

    circs = virtualizer.instantiations()

    qregs, circ_lists = zip(*circs.items())
    with Pool() as p:
        logging.info("Running circuits")
        now = time.time()
        all_results = p.starmap(
            _run_circuits, [(circuits, shots) for circuits in circ_lists]
        )
        logging.info(f"Running circuits took {time.time() - now} seconds")
        for qreg, results in zip(qregs, all_results):
            virtualizer.put_results(qreg, results)
        logging.info("Knitting results")
        now = time.time()
        final_result = virtualizer.knit(p)
        logging.info(f"Knitting took {time.time() - now} seconds")
    return final_result
