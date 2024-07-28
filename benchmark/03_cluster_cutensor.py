import time

import cotengra as ctg
from qiskit.circuit import QuantumCircuit
from cuquantum import CircuitToEinsum, contract, contract_path
import cupy as cp

import qtpu
from qtpu.compiler.terminators import reach_num_qubits
from qtpu.compiler.success import success_reach_qubits
from qtpu.evaluate import evaluate_estimator
from qiskit.primitives import BackendEstimator
from qiskit_aer.primitives import Estimator

# from qiskit.primitives import Estimator as AerEstimator

from benchmark.benchmarks import cluster_ansatz, linear_ansatz, brick_ansatz
from benchmark.util import contraction_cost_log10, append_to_csv


def execute_cutensornet(circuit: QuantumCircuit) -> dict[str, float]:
    myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
    pauli_string = "Z" * circuit.num_qubits
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)
    start = time.perf_counter()
    cost = contract_path(expression, *operands)[1].opt_cost
    compile_time = time.perf_counter() - start
    print(f"CutensorNet Compile Time: {compile_time}")

    # start = time.perf_counter()
    # ctg_cost = cotengra_cost(expression, operands)
    # ctg_compile_time = time.perf_counter() - start
    # logging.info(f"Cotengra Compile Time: {ctg_compile_time}")

    print(f"Cutensor Cost: {cost}")
    start = time.perf_counter()
    _ = contract(expression, *operands)
    runtime = time.perf_counter() - start
    print(f"Cutensor Runtime: {runtime}")
    return {
        "cutensor_cost": cost,
        # "ctg_cost": ctg_cost,
        "cutensor_runtime": runtime,
    }


def execute_qtpu(circuit: QuantumCircuit, max_cost: int) -> dict[str, float]:
    circuit.measure_all()
    start = time.perf_counter()
    htn = qtpu.cut(
        circuit,
        max_cost=max_cost,
        terminate_fn=reach_num_qubits(20),
        success_fn=success_reach_qubits(20),
        show_progress_bar=True,
        compression_methods=["qubits"],
        n_trials=100,
    )
    compile_time = time.perf_counter() - start
    print(f"QTPU Compile Time: {compile_time}")
    cost = 10 ** contraction_cost_log10(htn)
    print(f"QTPU Cost: {cost}")

    print([qt.circuit.num_qubits for qt in htn.quantum_tensors])

    start = time.perf_counter()
    for qt in htn.quantum_tensors:
        qt.generate_instances()
    gen_time = time.perf_counter() - start
    print(f"QTPU Generation Time: {gen_time}")

    estimator = Estimator(
        run_options={"method": "statevector", "shots": None}, approximation=True
    )
    # estimator.set_options(device="CPU", cuStateVec_enable=False)

    start = time.perf_counter()
    eq, operands = qtpu.evaluate(htn, eval_fn=evaluate_estimator(estimator))

    operands = [cp.array(op, dtype=cp.float32) for op in operands]

    for op in operands:
        print(op.shape)

    eval_time = time.perf_counter() - start
    print(f"QTPU Evaluation Time: {eval_time}")

    _ = contract_path(eq, *operands)

    start = time.perf_counter()
    _ = contract(eq, *operands)
    contract_time = time.perf_counter() - start
    print(f"QTPU Contract Time: {contract_time}")
    return {
        "qtpu_cost": cost,
        "qtpu_compile_time": compile_time,
        "qtpu_gen_time": gen_time,
        "qtpu_evaltime": eval_time,
        "qtpu_contract_time": contract_time,
    }


def cotengra_cost(equation: str, operands: list[cp.ndarray]) -> float:
    return ctg.einsum_tree(equation, *[op.shape for op in operands]).contraction_cost()


if __name__ == "__main__":

    name = "brick"

    benches = [
        # (4, 18, 2),
        # (5, 20, 2),
        (5, 15, 2),
        # (5, 20, 7),
        # (5, 20, 8),
        # (5, 20, 4),
        # (5, 20, 5),
    ]

    for n, s, d in benches:
        match name:
            case "linear":
                circuit = linear_ansatz(n * s, d).decompose()
                benchname = f"linear_{n * s}_{d}"
            case "brick":
                circuit = brick_ansatz(n * s, d)
                benchname = f"brick_{n * s}_{d}"
            case "cluster":
                circuit = cluster_ansatz(n * [s], d, seed=123)
                benchname = f"cluster_{n}_{s}_{d}"

        cu_res = execute_cutensornet(circuit)
        our_res = execute_qtpu(circuit, max_cost=1e12)

        print(
            f"results/cutensor.csv",
            {
                "name": benchname,
                **cu_res,
                **our_res,
            },
        )
