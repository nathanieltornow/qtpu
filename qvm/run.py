import logging
from time import perf_counter
from dataclasses import dataclass
from multiprocessing import Pool

from qiskit.providers import BackendV2
from qiskit.circuit import QuantumRegister as Fragment

from qvm.virtual_circuit import VirtualCircuit, generate_instantiations
from qvm.qvm_runner import QVMBackendRunner


logger = logging.getLogger("qvm")


@dataclass
class RunTimeInfo:
    run_time: float
    knit_time: float


def run_virtualizer(
    virt: VirtualCircuit, runner: QVMBackendRunner, backend: BackendV2 | None = None
) -> tuple[dict[int, float], RunTimeInfo]:
    jobs: dict[Fragment, str] = {}

    logger.info(
        f"Running virtualizer with {len(virt.fragment_circuits)} "
        + f"{tuple(circ.num_qubits for circ in virt.fragment_circuits.values())} "
        + f"fragments and {len(virt._vgate_instrs)} vgates..."
    )

    num_instances = 0
    now = perf_counter()
    for frag, frag_circuit in virt.fragment_circuits.items():
        instance_labels = virt.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circuit, instance_labels)
        num_instances += len(instantiations)
        jobs[frag] = runner.run(instantiations, backend)

    logger.info(f"Running {num_instances} instances...")
    results = {}
    for frag, job_id in jobs.items():
        results[frag] = runner.get_results(job_id)

    run_time = perf_counter() - now

    logger.info(f"Knitting...")

    with Pool(processes=8) as pool:
        now = perf_counter()
        res_dist = virt.knit(results, pool)
        knit_time = perf_counter() - now
        
    logger.info(f"Knitted in {knit_time:.2f}s.")

    return res_dist.nearest_probability_distribution(), RunTimeInfo(run_time, knit_time)
