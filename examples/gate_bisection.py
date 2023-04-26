import logging

from multiprocessing.pool import Pool

import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer import AerSimulator
from qiskit.quantum_info import hellinger_fidelity

import qvm
from _example_circuit import example_circuit


SHOTS = 10000


if __name__ == "__main__":
    logger = logging.getLogger("qvm")
    logger.setLevel(logging.INFO)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    logger.info("Logging level set to INFO.")

    circuit = example_circuit()

    # create a circuit with virtual gates
    # (virtual gates are denoted as a Barrier)
    virt_circuit = qvm.cut(circuit, technique="gate_bisection", num_fragments=2)

    # get a virtualizer
    virt = qvm.TwoFragmentGateVirtualizer(virt_circuit)
    frag_circs = virt.fragments()

    simulator = AerSimulator()
    results = {}
    for fragment, args in virt.instantiate().items():
        circuits = [qvm.insert_placeholders(frag_circs[fragment], arg) for arg in args]
        counts = simulator.run(circuits, shots=SHOTS).result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        results[fragment] = [qvm.QuasiDistr.from_counts(count) for count in counts]

    with Pool() as pool:
        res_distr = virt.knit(results, pool=pool)

    res_counts = res_distr.to_counts(SHOTS)
    perf_res_counts = AerSimulator().run(circuit, shots=SHOTS).result().get_counts()
    print(f"hellinger fidelity: {hellinger_fidelity(res_counts, perf_res_counts)}")
