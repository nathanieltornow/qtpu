# import logging
# from dataclasses import asdict
# from bench_generation import (
#     Benchmark,
#     generate_vqr_benchmarks,
#     generate_gen_bisection_benchmarks,
# )
# from tqdm import tqdm

# from qvm.virtualizer import Virtualizer
# from qvm.qvm_runner import QVMBackendRunner, IBMBackendRunner, LocalBackendRunner

# from _util import append_to_csv_file
# from _run_experiment import get_circuit_properties, run_experiment


# logger = logging.getLogger("qvm")
# logger.setLevel(logging.INFO)


# def run_benchmark(benches: list[Benchmark], runner: QVMBackendRunner):
#     progress = tqdm(total=sum(len(b.circuits) for b in benches))
#     progress.set_description("Running Benchmarks")

#     logger.info("Running Benchmarks")
#     for bench in benches:
#         for circ in bench.circuits:
#             cut_circ = bench.virt_compiler.run(circ)
#             virt = Virtualizer(cut_circ)
#             run_experiment(
#                 bench.result_file,
#                 circ,
#                 virt,
#                 runner,
#                 bench.backend,
#             )
#             progress.update(1)


# if __name__ == "__main__":
#     from qiskit.providers.fake_provider import FakeMontrealV2

#     from qiskit_ibm_runtime import QiskitRuntimeService

#     # Save your credentials on disk.

#     service = QiskitRuntimeService()

#     backend = service.get_backend("ibmq_kolkata")

#     # benches = generate_gen_bisection_benchmarks(
#     #     "qaoa", [.1, .2, .3], backend, num_vgates=4, reverse_order=False
#     # )
#     benches = generate_vqr_benchmarks("2local", [2], backend, num_vgates=2)

#     runner = IBMBackendRunner(service=service)

#     run_benchmark(benches, runner)
