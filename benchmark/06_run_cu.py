from time import perf_counter

from qiskit.circuit import QuantumCircuit
from qiskit_aer.primitives import Sampler

from qtpu.compiler.compiler import compile_reach_size
from qtpu.evaluate import evaluate_sampler

from benchmark.exec_ckt import ckt_execute_dummy, cut_ckt
from benchmark.exec_qtpu import qtpu_execute_dummy, qtpu_execute_dummy_cutensor
from benchmark.ansatz import generate_ansatz, qaoa1, qaoa2
from benchmark.util import append_to_csv, DummySampler, get_info


# benchmarks = {
#     # "qaoa2-2": qaoa2(2, 10, 3),
#     # "qaoa2-3": qaoa2(4, 10, 3),
#     # "qaoa2-4": qaoa2(4, 10, 4),
#     # "qaoa2-4": qaoa2(4, 10, 2),
#     # "qaoa2-5": qaoa2(5, 10, 2),
#     # "qaoa2-6": qaoa2(6, 10, 2),
#     # "qml-20": generate_ansatz("zz", 20, 2),
#     "vqe-100": generate_ansatz("linear", 100, 3),
#     "vqe-400": generate_ansatz("linear", 100, 4),
#     "vqe-800": generate_ansatz("linear", 100, 5),
#     # "qml-30": generate_ansatz("zz", 30, 2),
#     # "vqe-30": generate_ansatz("linear", 30, 2),
# }


QPU_SIZE = 15


benches = {
    "vqe": [generate_ansatz("linear", n, 2) for n in range(100, 101, 20)],
    # "qaoa2": [qaoa2(n, 10, 2) for n in range(6, 11)],
}

CSV = "runtime1.csv"

for name, bench_list in benches.items():

    for bench in bench_list:
        start = perf_counter()
        cut_circ = compile_reach_size(bench, QPU_SIZE, show_progress_bar=True)
        compile_time = perf_counter() - start

        print(get_info(cut_circ))

        qtpu_res = qtpu_execute_dummy(cut_circ)
        print(qtpu_res)
        qtpu_gpu_res = qtpu_execute_dummy_cutensor(cut_circ)
        print(qtpu_gpu_res)
        ckt_res = ckt_execute_dummy(cut_circ, 100000)
        print(ckt_res)

        append_to_csv(
            CSV,
            {"name": name, **get_info(cut_circ), **qtpu_res, **ckt_res, **qtpu_gpu_res},
        )
