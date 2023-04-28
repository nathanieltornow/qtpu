import logging


from qvm.util import fragment_circuit

from _example_circuit import example_circuit

import qvm


if __name__ == "__main__":
    logger = logging.getLogger("qvm")
    logger.setLevel(logging.INFO)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    circuit = example_circuit()

    virt_circuit = qvm.cut(
        circuit,
        technique="optimal",
        num_fragments=2,
        max_wire_cuts=2,
        max_gate_cuts=2,
        max_fragment_size=4,
    )
    print(fragment_circuit(virt_circuit))
