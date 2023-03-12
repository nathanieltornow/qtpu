import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit, QuantumRegister
from ray.util.multiprocessing import Pool

from qvm.cut_library.decomposition import bisect
from qvm.cut_library.util import fragment_circuit
from qvm.quasi_distr import QuasiDistr
from qvm.runtime.virtualizer import Virtualizer
from qvm.types import QPU, SampleMetaData


@dataclass
class RuntimeConfig:
    max_virt_gates: int
    max_circuit_size: int


def _run_on_qpu(
    qpu: QPU, circuits: list[QuantumCircuit], metadata: SampleMetaData | None = None
) -> list[QuasiDistr]:
    return qpu.sample(circuits, SampleMetaData(10000))


class QVMRuntime:
    def __init__(self, qpus: set[QPU]) -> None:
        self._qpus = qpus
        self._qpu_executor = ThreadPoolExecutor()
        self._virtualize_pool = Pool()

    def sample(self, circuit: QuantumCircuit) -> QuasiDistr:
        circuit = bisect(circuit)
        return self._run_virtualized_circuit(circuit)

    def _run_virtualized_circuit(
        self,
        circuit: QuantumCircuit,
    ) -> QuasiDistr:
        logging.info(f"Running virtualized circuit with {len(circuit.qregs)} fragments")
        logging.info(circuit)

        virtualizer = Virtualizer(circuit)
        circs = virtualizer.instantiations()

        qpus_list = list(self._qpus)

        qregs, circ_lists = zip(*circs.items())
        with Pool() as p:
            logging.info(f"Running {sum(len(circs) for circs in circ_lists)} circuits")
            now = time.time()
            all_results = p.starmap(
                _run_on_qpu,
                [
                    (qpus_list[i % len(qpus_list)], circuits, None)
                    for i, circuits in enumerate(circ_lists)
                ],
            )
            logging.info(f"Running circuits took {time.time() - now} seconds")
            for qreg, results in zip(qregs, all_results):
                virtualizer.put_results(qreg, results)
            logging.info("Knitting results")
            now = time.time()
            final_result = virtualizer.knit(p)
            logging.info(f"Knitting took {time.time() - now} seconds")
        return final_result
