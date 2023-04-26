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
    virt_circuit = qvm.cut(
        circuit,
        technique="optimal",
        num_fragments=2,
        max_wire_cuts=2,
        max_gate_cuts=2,
        max_fragment_size=5,
    )
    print(virt_circuit)
