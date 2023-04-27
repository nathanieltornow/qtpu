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
