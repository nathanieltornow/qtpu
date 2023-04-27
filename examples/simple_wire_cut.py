import logging
from multiprocessing.pool import Pool

import numpy as np
from _example_circuit import example_circuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

import qvm

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

    virt_circuit = qvm.cut(
        circuit, technique="wire_optimal", num_fragments=2, max_cuts=1
    )

    print(virt_circuit)

        # get a virtualizer
    virt = qvm.SingleWireVirtualizer(virt_circuit)
    frag_circs = virt.fragments()
    print(frag_circs)
    # for frag, circ in frag_circs.items():
    #     # print(frag)
    #     print(circ)

    simulator = AerSimulator()
    results = {}
    for fragment, args in virt.instantiate().items():
        circuits = [qvm.insert_placeholders(frag_circs[fragment], arg) for arg in args]
        for circ in circuits:
            print(circ)
        counts = simulator.run(circuits, shots=SHOTS).result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        results[fragment] = [qvm.QuasiDistr.from_counts(count) for count in counts]

    with Pool() as pool:
        res_distr = virt.knit(results, pool=pool)

    res_counts = res_distr.to_counts(SHOTS)
    print(res_counts)
    perf_res_counts = AerSimulator().run(circuit, shots=SHOTS).result().get_counts()
    print(perf_res_counts)
    print(f"hellinger fidelity: {hellinger_fidelity(res_counts, perf_res_counts)}")