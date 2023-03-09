import time
from concurrent.futures import ThreadPoolExecutor

from ray.util.multiprocessing import Pool
from qiskit.circuit import QuantumCircuit
from qiskit.providers.ibmq import IBMQBackend
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.compiler import transpile

from qvm.cut_library.util import fragment_circuit
from qvm.quasi_distr import QuasiDistr
from qvm.runtime.virtualizer import Virtualizer

OPTIMIZATION_LEVEL = 0


def _run_circuits(
    circuits: list[QuantumCircuit], backend: IBMQBackend, shots: int = 10000
) -> list[QuasiDistr]:
    circuits = transpile(
        circuits, backend=backend, optimization_level=OPTIMIZATION_LEVEL
    )
    manager = IBMQJobManager()
    results = manager.run(circuits, backend=backend, shots=shots).results()
    counts = [results.get_counts(i) for i in range(len(circuits))]
    return [QuasiDistr.from_counts(counts=count, shots=shots) for count in counts]


def sample_on_ibmq_backend(
    virtual_circuit: QuantumCircuit, backend: IBMQBackend, shots: int = 10000
) -> QuasiDistr:
    frag_circ = fragment_circuit(virtual_circuit)
    virtualizer = Virtualizer(frag_circ)
    instances = virtualizer.instantiations()
    qregs, circ_lists = zip(*instances.items())
    print(
        f"Running {sum(len(circs) for circs in circ_lists)} circuits with maximum circuit size of {max(len(qreg) for qreg in qregs)} qubits"
    )
    with ThreadPoolExecutor(len(qregs)) as circ_exec:
        now = time.perf_counter()
        all_results = circ_exec.map(
            _run_circuits, circ_lists, [backend] * len(qregs), [shots] * len(qregs)
        )
    print(f"Running circuits took {time.perf_counter() - now} seconds")
    for qreg, results in zip(qregs, all_results):
        virtualizer.put_results(qreg, results)
    print("Knitting results")
    with Pool() as knit_exec:
        now = time.perf_counter()
        final_result = virtualizer.knit(knit_exec)
        print(f"Knitting took {time.perf_counter() - now} seconds")
    return final_result
