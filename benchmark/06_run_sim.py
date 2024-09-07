from time import perf_counter

from qiskit.circuit import QuantumCircuit
from qiskit_aer.primitives import Sampler

from qtpu.compiler.compiler import compile_reach_size
from qtpu.evaluate import evaluate_sampler
from qtpu.circuit import cuts_to_moves, circuit_to_hybrid_tn

from benchmark.exec_ckt import ckt_execute_dummy, cut_ckt
from benchmark.exec_qtpu import (
    qtpu_execute_dummy,
    qtpu_execute_dummy_cutensor,
    qtpu_execute_cutensor,
    run_circuit_cutensor,
)
from benchmark.ansatz import generate_ansatz, qaoa1, qaoa2
from benchmark.util import append_to_csv, DummySampler, get_info


from cuquantum import Network, CircuitToEinsum, contract, contract_path
import cupy as cp


def run_cutensor(circuit: QuantumCircuit):
    myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
    pauli_string = "Z" * circuit.num_qubits
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)

    with Network(expression, *operands) as tn:
        start = perf_counter()
        path, info = tn.contract_path()
        tn.autotune(iterations=5)
        compile_time = perf_counter() - start

        print(info)

        start = perf_counter()
        result = tn.contract()
        exec_time = perf_counter() - start

    return result, {
        "cutensor_comp": compile_time,
        "cutensor_run": exec_time,
        "cutensor_cost": info.opt_cost,
    }


QPU_SIZE = 25


benches = {
    "qaoa1": [qaoa2(6, n, 1) for n in range(25, 28)],
}

CSV = "runtime_sim2ß.csv"


from qiskit.primitives import BackendEstimator
from qiskit_aer import AerSimulator
from qtpu.evaluate import evaluate_estimator

for name, bench_list in benches.items():

    for bench in bench_list:
        bench.measure_all()

        start = perf_counter()
        cut_circ = compile_reach_size(bench, QPU_SIZE, show_progress_bar=True)
        compile_time = perf_counter() - start

        cut_circ = cuts_to_moves(cut_circ)
        print(get_info(cut_circ))

        sim = AerSimulator()
        sim.set_options(device="GPU")

        est = BackendEstimator(sim)

        qtpu_res = qtpu_execute_cutensor(
            cut_circ,
            0.0,
            # eval_fn=lambda circuits: [run_circuit_cutensor(c) for c in circuits],
            # eval_fn=lambda circuits: est.run(circuits, shots=1000).result().values,
            eval_fn=evaluate_estimator(est),
        )
        print(qtpu_res)

        cutensor_res = run_cutensor(bench)[1]

        append_to_csv(
            CSV,
            {
                "name": name,
                "qtpu_compile": compile_time,
                **qtpu_res,
                **cutensor_res,
                **get_info(cut_circ),
            },
        )
