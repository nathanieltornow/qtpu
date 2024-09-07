from time import perf_counter
from typing import Callable

import numpy as np

from qiskit.circuit import QuantumCircuit

from qtpu.contract import evaluate_hybrid_tn
from qtpu.circuit import circuit_to_hybrid_tn, cuts_to_moves
from qtpu.helpers import defer_mid_measurements


def qtpu_execute_dummy(
    circuit: QuantumCircuit, tolerance: float = 0.0
) -> dict[str, float]:

    start = perf_counter()
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit)
    if tolerance > 0:
        htn.simplify(tolerance)
    prep_time = perf_counter() - start

    tn = htn.to_tensor_network()
    start = perf_counter()
    res = tn.contract(optimize="auto", output_inds=[])
    post_time = perf_counter() - start

    return {"qtpu_pre": prep_time, "qtpu_post": post_time}


def qtpu_execute_dummy_cutensor(circuit: QuantumCircuit, tolerance: float = 0.0):
    from cuquantum import Network, CircuitToEinsum, contract, contract_path
    import cupy as cp

    start = perf_counter()
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit)

    htn.simplify(tolerance)
    prep_time = perf_counter() - start

    tn = htn.to_tensor_network()

    eq, tensors = tn.get_equation(), tn.tensors
    operands = [cp.array(t.data, dtype=cp.float32) for t in tensors]

    with Network(eq, *operands) as cutn:
        start = perf_counter()
        path, info = cutn.contract_path()
        cutn.autotune(iterations=5)
        compile_time = perf_counter() - start

        start = perf_counter()
        result = cutn.contract()
        exec_time = perf_counter() - start

    return {
        "qtpu_gpu_pre": prep_time,
        "qtpu_gpu_comp": compile_time,
        "qtpu_gpu_post": exec_time,
    }


def run_qtpu(
    circuit: QuantumCircuit,
    tolerance: float = 0.0,
    eval_fn: Callable[[list[QuantumCircuit]], list] | None = None,
) -> tuple[float, dict]:

    start = perf_counter()
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit)
    htn.simplify(tolerance)

    for qt in htn.quantum_tensors:
        qt.generate_instances()

    preptime = perf_counter() - start

    start = perf_counter()
    tn = evaluate_hybrid_tn(htn, eval_fn)
    runtime = perf_counter() - start

    start = perf_counter()
    res = tn.contract(optimize="auto", output_inds=[])
    posttime = perf_counter() - start

    return res, {
        "qtpu_pre": preptime,
        "qtpu_run": runtime,
        "qtpu_post": posttime,
    }


def qtpu_execute_cutensor(
    circuit: QuantumCircuit,
    tolerance: float = 0.0,
    eval_fn: Callable[[list[QuantumCircuit]], list] | None = None,
    return_result: bool = False,
) -> tuple[float, dict]:

    from cuquantum import Network
    import cupy as cp

    start = perf_counter()
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit)
    htn.simplify(tolerance)
    num_circuits = htn.num_circuits()

    for qt in htn.quantum_tensors:
        qt.generate_instances()

    preptime = perf_counter() - start

    start = perf_counter()
    tn = evaluate_hybrid_tn(htn, eval_fn)
    runtime = perf_counter() - start

    eq, tensors = tn.get_equation(), tn.tensors
    operands = [cp.array(t.data, dtype=cp.float32) for t in tensors]

    with Network(eq, *operands) as cutn:
        start = perf_counter()
        path, info = cutn.contract_path()
        cutn.autotune(iterations=5)
        compile_time = perf_counter() - start

        start = perf_counter()
        result = cutn.contract()
        exec_time = perf_counter() - start

    if return_result:
        return result, {
            "qtpu_gpu_pre": preptime,
            "qtpu_gpu_comp": compile_time,
            "qtpu_gpu_run": runtime,
            "qtpu_gpu_post": exec_time,
            "num_instances": num_circuits,
        }

    return {
        "qtpu_gpu_pre": preptime,
        "qtpu_gpu_comp": compile_time,
        "qtpu_gpu_run": runtime,
        "qtpu_gpu_post": exec_time,
        "num_instances": num_circuits
    }


def run_circuit_cutensor(circuit: QuantumCircuit) -> float:
    from cuquantum import Network, CircuitToEinsum
    import cupy as cp

    circuit = defer_mid_measurements(circuit)
    myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)

    obs = _get_Z_observable(circuit)

    expression, operands = myconverter.expectation(obs, lightcone=True)

    with Network(expression, *operands) as tn:
        path, info = tn.contract_path()
        tn.autotune(iterations=5)
        result = tn.contract()

    return float(np.real(result))


def _get_meas_qubits(circuit: QuantumCircuit) -> list[int]:
    measured_qubits = sorted(
        set(
            circuit.qubits.index(instr.qubits[0])
            for instr in circuit
            if instr.operation.name == "measure"
        ),
    )
    return measured_qubits


def _get_Z_observable(circuit: QuantumCircuit) -> str:
    measured_qubits = _get_meas_qubits(circuit)
    obs = ["I"] * circuit.num_qubits
    for qubit in measured_qubits:
        obs[qubit] = "Z"
    return "".join(obs)
