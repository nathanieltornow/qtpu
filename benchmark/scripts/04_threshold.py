import numpy as np

from qiskit.circuit import QuantumCircuit

from qtpu.compiler.compiler import compile_reach_size

from benchmark.util import get_info, concat_data
from benchmark.ansatz import qaoa2
from benchmark.exec_ckt import ckt_execute_dummy, cut_ckt
from benchmark.exec_qtpu import run_cutensor, qtpu_execute_dummy_cutensor
from qtpu.helpers import sample_quimb


def get_benches():
    benches = [qaoa2(5, n, 1, 2) for n in [22, 23]]
    return benches


circuit = QuantumCircuit.from_qasm_file(
    "benchmark/qasm/qasmbench-large/qugan_n71/qugan_n71.qasm"
)


# samples = sample_quimb(circuit, 10000)
# print(samples)
#
# from cuquantum import Network, CircuitToEinsum
# from time import perf_counter
# import cupy as cp

# myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
# pauli_string = "Z" * circuit.num_qubits
# expression, operands = myconverter.expectation(pauli_string, lightcone=True)

# with Network(expression, *operands) as tn:
#     start = perf_counter()
#     path, info = tn.contract_path()

#     tn.autotune(iterations=5)
#     compile_time = perf_counter() - start

#     print(info)

#     start = perf_counter()
#     result = tn.contract()
#     exec_time = perf_counter() - start
# from qiskit import transpile

res, cutensortimes = run_cutensor(circuit)

# cut_ckt(transpile(circuit, basis_gates=["cx", "rz", "ry", "measure"]), 19)
cut_circ = compile_reach_size(
    circuit, 30, n_trials=20, show_progress_bar=True, max_cost=1e15
)
info = get_info(cut_circ)
print(res)
print(cutensortimes)
print(info)
exit()

# circ, meta = get_benches()[0]
# cut_circ = compile_reach_size(circ, meta["n"])

# ckt_times = ckt_execute_dummy(cut_circ)
# qtpu_gpu_times = qtpu_execute_dummy_cutensor(cut_circ)

for bench in get_benches():
    circ, meta = bench

    _, cutensortimes = run_cutensor(circ)

    cut_circ = compile_reach_size(circ, meta["n"])
    info = get_info(cut_circ)

    data = {
        **meta,
        **info,
        **cutensortimes,
        # "num_samples": num_samples,
    }

    # if num_samples <= 1e5:

    # data = {**data, **ckt_times}

    # data = {**data, **qtpu_gpu_times}

    concat_data("benchmark/results/threshold2.json", data)
