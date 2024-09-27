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

from benchmark.exec_qtpu import qtpu_execute, qtpu_execute_cutensor, run_cutensor
from benchmark.exec_ckt import ckt_execute, ckt_execute_dummy

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError


sim = AerSimulator(device="GPU", cuStateVec_enable=True)
estimator = BackendEstimatorV2(backend=sim)
sampler = BackendSamplerV2(backend=sim)


def run_sv(circuit: QuantumCircuit) -> float:
    start = perf_counter()
    estimator.run([(circuit, "Z" * circuit.num_qubits)]).result()
    end = perf_counter()
    return end - start


def main():

    num_samples = np.inf

    sim = AerSimulator(device="GPU")
    estimator = BackendEstimatorV2(backend=sim)

    for n in [150, 200]:
        circuit = qaoa2(n // 10, 10, 1, 2)

        res1, cutensor_times = run_cutensor(circuit)
        print(cutensor_times["cutensor_compile"] + cutensor_times["cutensor_exec"])

        try:
            with ThreadPoolExecutor(1) as executor:
                future = executor.submit(run_sv, circuit)
                sv_time = future.result(timeout=600)
        except Exception as e:
            print(e)
            sv_time = -1.0

        print(sv_time)
        cut_circuit = compile_reach_size(
            circuit, 10, n_trials=40, show_progress_bar=True
        )

        info = get_info(cut_circuit)
        print(info)

        _, gpu_times = qtpu_execute_cutensor(
            cut_circuit, estimator, num_samples=num_samples
        )
        print(gpu_times)
        # #

        try:
            with ThreadPoolExecutor(1) as executor:
                future = executor.submit(ckt_execute_dummy, cut_circuit, 100000)
                ckt_times = future.result(timeout=1200)
        except Exception as e:
            print(e)
            ckt_times = {
                "ckt_pre": -1.0,
                "ckt_run": -1.0,
                "ckt_post": -1.0,
                "ckt_num_instances": -1.0,
            }
        print(ckt_times)

        append_to_csv(
            "benchmark/results/simulation_full2.csv",
            {
                "num_qubits": n,
                "num_samples": num_samples,
                "qtpu_num_instances": info["num_instances"],
                "qtpu_depth": info["depth"],
                "sv_time": sv_time,
                **gpu_times,
                **ckt_times,
                **cutensor_times,
            },
        )


if __name__ == "__main__":
    main()
