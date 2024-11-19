from concurrent.futures import ThreadPoolExecutor, TimeoutError

from qtpu.compiler._compiler import compile_reach_size

from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2

from benchmark.util import get_info, concat_data
from benchmark.ansatz import vqe, qml, qaoa1, qaoa2
from benchmark.exec_ckt import ckt_execute
from benchmark.exec_qtpu import qtpu_execute_cutensor, run_cutensor


file_path = "benchmark/results/end_to_end_estimator.json"


def get_benches():
    benches = []
    benches += [vqe(n, 2) for n in range(20, 201, 20)]
    benches += [qml(n, 2) for n in range(20, 201, 20)]

    benches += [qaoa1(n, 10, 2, 1) for n in range(2, 21, 2)]
    benches += [qaoa2(n, 10, 2, 1) for n in range(2, 21, 2)]
    # benches += [vqe(n, 1) for n in range(20, 61, 10)]
    # benches += [qml(n, 1) for n in range(20, 61, 10)]
    # benches += [qaoa1(n, 10, 1, 1) for n in range(2, 7, 1)]
    # benches += [qaoa2(n, 10, 1, 1) for n in range(2, 7, 1)]
    return benches


sim = AerSimulator(device="GPU", shots=20000)
sampler = BackendSamplerV2(backend=sim)
estimator = BackendEstimatorV2(backend=sim)

NUM_SAMPLES = int(1e4)


def run_bench(circ, meta):
    cut_circ = compile_reach_size(circ, 15)
    info = get_info(cut_circ)
    print(info)

    cutensor_res, cutensor_times = run_cutensor(circ)

    qtpu_res, qtpu_gpu_times = qtpu_execute_cutensor(
        cut_circ.measure_all(inplace=False), estimator, num_samples=NUM_SAMPLES
    )

    # if meta["reps"] == 1 and meta.get("m", 1) == 1:
    ckt_res, ckt_times = ckt_execute(cut_circ, sampler, num_samples=NUM_SAMPLES)
    data = {
        **meta,
        **info,
        **cutensor_times,
        **ckt_times,
        **qtpu_gpu_times,
        "cutensor_res": cutensor_res,
        "ckt_res": ckt_res,
        "qtpu_res": qtpu_res,
        "num_samples": NUM_SAMPLES,
    }
    # else:
    #     data = {
    #         **meta,
    #         **info,
    #         **cutensor_times,
    #         **qtpu_gpu_times,
    #         "cutensor_res": cutensor_res,
    #         "qtpu_res": qtpu_res,
    #     }

    concat_data(file_path, data)


for bench in get_benches():
    circ, meta = bench
    # set timeout to 10 minutes
    run_bench(circ, meta)
