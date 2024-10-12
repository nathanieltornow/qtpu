from qtpu.compiler.compiler import compile_reach_size

from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2

from benchmark.util import get_info, concat_data
from benchmark.ansatz import vqe, qml, qaoa1, qaoa2
from benchmark.exec_ckt import ckt_execute, ckt_numcoeffs
from benchmark.exec_qtpu import (
    qtpu_execute_cutensor,
    run_cutensor,
    qtpu_num_coeffs,
)


def get_benches():
    n = 40
    benches = [
        vqe(40, 2),
        qml(40, 2),
        qaoa1(n // 10, 10, 2, 1),
        qaoa2(n // 10, 10, 2, 1),
    ]
    return benches


sim = AerSimulator(device="GPU", shots=20000)
sampler = BackendSamplerV2(backend=sim)

for num_samples in [10000]:
    num_samples = int(num_samples)
    for bench in get_benches():

        circ, meta = bench

        perf_res, cutensortimes = run_cutensor(circ)

        cut_circ = compile_reach_size(circ, 15)

        info = get_info(cut_circ)

        qtpu_res, qtpu_gpu_times = qtpu_execute_cutensor(
            cut_circ.measure_all(inplace=False), sim, num_samples
        )
        print(qtpu_res, perf_res)

        ckt_res, ckt_times = ckt_execute(cut_circ, sampler, num_samples)

        data = {
            **meta,
            **info,
            **ckt_times,
            **qtpu_gpu_times,
            **cutensortimes,
            **ckt_times,
            "num_samples": num_samples,
            "ckt_coeffs": ckt_numcoeffs(cut_circ, num_samples),
            "qtpu_coeffs": qtpu_num_coeffs(cut_circ, num_samples),
            "perf_res": perf_res,
            "ckt_res": ckt_res,
            "qtpu_res": qtpu_res,
        }
        concat_data("benchmark/results/sampling3.json", data)
