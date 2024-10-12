from qtpu.compiler.compiler import compile_reach_size

from benchmark.util import get_info, concat_data
from benchmark.ansatz import vqe, qml, qaoa1, qaoa2
from benchmark.exec_ckt import ckt_execute_dummy
from benchmark.exec_qtpu import qtpu_execute_dummy, qtpu_execute_dummy_cutensor


file_path = "benchmark/results/postprocessing.json"


def get_benches():
    benches = []
    benches += [vqe(n, 2) for n in range(20, 101, 20)]
    benches += [qml(n, 2) for n in range(20, 101, 20)]
    benches += [qaoa1(n, 10, 2, 1) for n in range(2, 11, 2)]
    benches += [qaoa2(n, 10, 2, 1) for n in range(2, 11, 2)]
    return benches


for bench in get_benches():
    circ, meta = bench
    cut_circ = compile_reach_size(circ, 15)
    del circ

    info = get_info(cut_circ)
    
    ckt_times = ckt_execute_dummy(cut_circ, 1e5)
    qtpu_times = qtpu_execute_dummy(cut_circ)
    qtpu_gpu_times = qtpu_execute_dummy_cutensor(cut_circ)

    data = {
        **meta,
        **info,
        **ckt_times,
        **qtpu_times,
        **qtpu_gpu_times,
    }
    concat_data(file_path, data)
