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


def estimate_runtime(circuit: QuantumCircuit, num_shots: int):
    htn = circuit_to_hybrid_tn(circuit)
    htn.simplify(0.05)
    time = 0
    for qt in htn.quantum_tensors:
        depth = qt._circuit.depth(lambda instr: len(instr.qubits) > 1)
        n_inst = qt.ind_tensor.size
        time += depth * 100e-9 * n_inst * num_shots
    return time


QPU_SIZE = 25


benches = {
    "qml": [generate_ansatz("linear", 100, n) for n in range(2, 6)],
    # "qaoa2": [qaoa2(6, n, 2) for n in range(18, 26)],
}

CSV = "runtime_sim4.csv"


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

        qtpu_res = qtpu_execute_dummy_cutensor(
            cut_circ,
            0.05,
        )

        cutensor_res = run_cutensor(bench)[1]

        append_to_csv(
            CSV,
            {
                "name": name,
                "qtpu_compile": compile_time,
                "qtpu_runtime": estimate_runtime(cut_circ, 10000),
                **qtpu_res,
                **cutensor_res,
                **get_info(cut_circ),
            },
        )
