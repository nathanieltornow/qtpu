from time import perf_counter

import cotengra as ctg
from qiskit.circuit import QuantumCircuit
from cuquantum import Network, CircuitToEinsum, contract, contract_path
import cupy as cp

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2
from qtpu.circuit import circuit_to_hybrid_tn
from qtpu.compiler.compiler import compile_reach_size
from benchmark.ansatz import qaoa2, generate_ansatz

from benchmark.util import append_to_csv

from benchmark.exec_qtpu import qtpu_execute, qtpu_execute_cutensor, run_cutensor
from benchmark.exec_ckt import ckt_execute


def main():
    # circuit = qaoa2(4, 10, 2)
    circuit = generate_ansatz("linear", 40, 2)


    # _, times = run_cutensor(circuit)
    # print(times["cutensor_compile"] + times["cutensor_exec"])

    # exit()

    cut_circuit = compile_reach_size(circuit, 10, n_trials=20, show_progress_bar=True)

    sim = AerSimulator(device="GPU")

    sampler = BackendSamplerV2(backend=sim)
    sampler.options.default_shots = 20000

    estimator = BackendEstimatorV2(backend=sim)
    estimator.options.default_shots = 20000

    for num_samples in [10000]:

        _, ckt_times = ckt_execute(cut_circuit, sampler, num_samples)
        print(ckt_times)

        qtpu_circuit = cut_circuit.measure_all(inplace=False)

        _, qtpu_times = qtpu_execute(qtpu_circuit, estimator, num_samples)
        print(qtpu_times)

        _, qtpu_gpu_times = qtpu_execute_cutensor(qtpu_circuit, estimator, num_samples)
        print(qtpu_gpu_times)

        

        append_to_csv(
            "benchmark/results/end_to_end_vqe.csv",
            {
                "num_samples": num_samples,
                **qtpu_times,
                **qtpu_gpu_times,
                **ckt_times,
            },
        )


if __name__ == "__main__":
    main()
