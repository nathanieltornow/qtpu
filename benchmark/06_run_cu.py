from time import perf_counter

from qiskit.circuit import QuantumCircuit
from qiskit_aer.primitives import Sampler

from qtpu.compiler.compiler import compile_reach_size
from qtpu.evaluate import evaluate_sampler

from benchmark.exec_ckt import ckt_execute_dummy, cut_ckt
from benchmark.exec_qtpu import qtpu_execute_dummy, qtpu_execute_dummy_cutensor
from benchmark.ansatz import generate_ansatz, qaoa1, qaoa2
from benchmark.util import append_to_csv, DummySampler, get_info


QPU_SIZE = 15


benches = {
    # "vqe": [generate_ansatz("linear", n, 2) for n in range(100, 101, 20)],
    # "qml": [generate_ansatz("zz", n, 2) for n in range(80, 201, 20)],
    "qoao1": [qaoa1(n, 10, 2) for n in range(2, 11)],
    # "qaoa2": [qaoa2(n, 10, 2) for n in range(1, 11)],
}

CSV = "results/runtime_new.csv"

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
            {
                "name": name,
                "num_qubits": bench.num_qubits,
                **qtpu_res,
                **ckt_res,
                **qtpu_gpu_res,
            },
        )
