from time import perf_counter

import cotengra as ctg
from qiskit.circuit import QuantumCircuit
from cuquantum import Network, CircuitToEinsum, contract, contract_path
import cupy as cp

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler, EstimatorV2
from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2
from qtpu.circuit import circuit_to_hybrid_tn
from qtpu.compiler.compiler import compile_reach_size
from benchmark.ansatz import qaoa2


from benchmark.util import append_to_csv, get_info

from benchmark.exec_qtpu import qtpu_execute, qtpu_execute_dummy_cutensor, run_cutensor
from benchmark.exec_ckt import ckt_execute, ckt_execute_dummy

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np


def main():

    num_samples = np.inf

    for n in [4, 5, 6, 7, 8, 9, 10, 11]:
        circuit = qaoa2(5, 10, 1, 2)

        res1, cutensor_times = run_cutensor(circuit)
        print(cutensor_times["cutensor_compile"] + cutensor_times["cutensor_exec"])

        # cut_circuit = compile_reach_size(
        #     circuit, n_qubits, n_trials=20, show_progress_bar=True
        # )

        # info = get_info(cut_circuit)

        # gpu_times = qtpu_execute_dummy_cutensor(cut_circuit, num_samples=num_samples)
        # print(gpu_times)
        # #
        # ckt_times = ckt_execute_dummy(cut_circuit, 100000)
        # print(ckt_times)

        # append_to_csv(
        #     "benchmark/results/threshold5.csv",
        #     {
        #         "num_qubits": n_qubits,
        #         "num_samples": num_samples,
        #         "qtpu_num_instances": info["num_instances"],
        #         "qtpu_depth": info["depth"],
        #         **gpu_times,
        #         **ckt_times,
        #         **cutensor_times,
        #     },
        # )


if __name__ == "__main__":
    main()
