
from qvm.compiler import MetisCutter

from circuits import get_circuits
from bench.util import get_virtual_circuit_info, append_to_csv

RESULTFILE = "results/knit_overhead.csv"

def run():
    circuit = get_circuits("vqe_2", (20, 21))[0]

    for num_fragments in [2, 3, 4, 5]:
        cutter = MetisCutter(num_fragments)
        cut_circuit = cutter.run(circuit)

        info = get_virtual_circuit_info(cut_circuit).to_dict()



