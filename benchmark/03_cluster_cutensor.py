import logging
import time

import cotengra as ctg
from qiskit.circuit import QuantumCircuit
from cuquantum import CircuitToEinsum, contract, contract_path
import cupy as cp

import qtpu
from qtpu.compiler.terminators import reach_num_qubits
from qtpu.compiler.success import success_reach_qubits

from benchmark.benchmarks import cluster_ansatz, linear_ansatz, brick_ansatz
from benchmark.util import contraction_cost_log10, append_to_csv


logging.basicConfig(level=logging.INFO)


def execute_cutensornet(circuit: QuantumCircuit) -> dict[str, float]:
    myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
    pauli_string = "Z" * circuit.num_qubits
    expression, operands = myconverter.expectation(pauli_string, lightcone=True)
    start = time.perf_counter()
    cost = contract_path(expression, *operands)[1].opt_cost
    compile_time = time.perf_counter() - start
    logging.info(f"CutensorNet Compile Time: {compile_time}")

    # start = time.perf_counter()
    # ctg_cost = cotengra_cost(expression, operands)
    # ctg_compile_time = time.perf_counter() - start
    # logging.info(f"Cotengra Compile Time: {ctg_compile_time}")

    logging.info(f"Cutensor Cost: {cost}")
    start = time.perf_counter()
    _ = contract(expression, *operands)
    runtime = time.perf_counter() - start
    logging.info(f"Cutensor Runtime: {runtime}")
    return {
        "cutensor_cost": cost,
        # "ctg_cost": ctg_cost,
        "cutensor_runtime": runtime,
    }


def execute_qtpu_fake(
    circuit: QuantumCircuit, max_cost: int, shots: int, parallel: bool = False
) -> dict[str, float]:
    start = time.perf_counter()
    htn = qtpu.cut(
        circuit,
        max_cost=1e15,
        terminate_fn=reach_num_qubits(20),
        success_fn=success_reach_qubits(20),
        show_progress_bar=True,
        n_trials=100,
    )
    compile_time = time.perf_counter() - start
    logging.info(f"QTPU Compile Time: {compile_time}")
    cost = 10 ** contraction_cost_log10(htn)
    logging.info(f"QTPU Cost: {cost}")

    circuit_times = [
        qt.circuit.decompose().depth() * 1e-8 * shots for qt in htn.quantum_tensors
    ]
    runtime = max(circuit_times) if parallel else sum(circuit_times)
    logging.info(f"QTPU Runtime: {runtime}")

    print([qt.circuit.num_qubits for qt in htn.quantum_tensors])

    eval_tensors = [
        cp.random.randn(*qtens.shape, dtype=cp.float64) for qtens in htn.quantum_tensors
    ]
    operands = eval_tensors + [
        cp.array(ct.data, dtype=cp.float64) for ct in htn.classical_tensors
    ]

    eq = htn.equation()
    _ = contract_path(eq, *operands)

    start = time.perf_counter()
    _ = contract(eq, *operands)
    contract_time = time.perf_counter() - start
    logging.info(f"QTPU Contract Time: {contract_time}")
    return {
        "qtpu_cost": cost,
        "qtpu_runtime": runtime,
        "qtpu_contract_time": contract_time,
    }


def cotengra_cost(equation: str, operands: list[cp.ndarray]) -> float:
    return ctg.einsum_tree(equation, *[op.shape for op in operands]).contraction_cost()


if __name__ == "__main__":

    name = "brick"

    benches = [
        # (4, 18, 2),
        # (5, 20, 2),
        # (5, 20, 3),
        (5, 20, 7),
        (5, 20, 8),
        # (5, 20, 4),
        # (5, 20, 5),
    ] * 5

    for n, s, d in benches:
        match name:
            case "linear":
                circuit = linear_ansatz(n * s, d)
                benchname = f"linear_{n * s}_{d}"
            case "brick":
                circuit = brick_ansatz(n * s, d)
                benchname = f"brick_{n * s}_{d}"
            case "cluster":
                circuit = cluster_ansatz(n * [s], d, seed=123)
                benchname = f"cluster_{n}_{s}_{d}"

        cu_res = execute_cutensornet(circuit)
        our_res = execute_qtpu_fake(circuit, 1000, 1000000)

        print(
            f"results/cutensor.csv",
            {
                "name": benchname,
                **cu_res,
                **our_res,
            },
        )
        logging.info(f"Finished cluster_{n}_{s}_{d}")
