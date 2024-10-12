from time import perf_counter

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler

from qtpu.contract import evaluate_hybrid_tn
from qtpu.circuit import circuit_to_hybrid_tn, cuts_to_moves
from qtpu.helpers import defer_mid_measurements


def qtpu_num_coeffs(circuit: QuantumCircuit, num_samples: int = np.inf) -> int:
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit, num_samples)
    return np.prod([tens.size for tens in htn.qpd_tensors])


def qtpu_execute_dummy(
    circuit: QuantumCircuit, num_samples: int = np.inf
) -> dict[str, float]:

    start = perf_counter()
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit, num_samples)

    prep_time = perf_counter() - start

    tn = htn.to_tensor_network()
    start = perf_counter()
    res = tn.contract(optimize="auto", output_inds=[])
    post_time = perf_counter() - start

    return {"qtpu_pre": prep_time, "qtpu_post": post_time}


def qtpu_execute_dummy_cutensor(
    circuit: QuantumCircuit, num_samples: int = np.inf
) -> dict[str, float]:
    from cuquantum import Network, CircuitToEinsum, contract, contract_path
    import cupy as cp

    start = perf_counter()
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit, num_samples)

    prep_time = perf_counter() - start

    tn = htn.to_tensor_network()

    eq, tensors = tn.get_equation(), tn.tensors
    print(eq)
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


def qtpu_execute(
    circuit: QuantumCircuit,
    evaluator: BaseEstimator | BaseSampler | None = None,
    num_samples: int = np.inf,
) -> tuple[float, dict]:
    circuit = circuit.copy()
    circuit.remove_final_measurements()
    circuit.measure_all()

    start = perf_counter()
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit, num_samples)

    for qt in htn.quantum_tensors:
        qt.generate_instances()

    preptime = perf_counter() - start

    start = perf_counter()
    tn = evaluate_hybrid_tn(htn, evaluator)
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
    evaluator: BaseEstimator | BaseSampler | None = None,
    num_samples: int = np.inf,
) -> tuple[float, dict]:

    from cuquantum import Network
    import cupy as cp

    start = perf_counter()
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit, num_samples)
    num_circuits = htn.num_circuits()

    for qt in htn.quantum_tensors:
        qt.generate_instances()

    preptime = perf_counter() - start

    start = perf_counter()
    tn = evaluate_hybrid_tn(htn, evaluator)
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

    return float(np.real(result)), {
        "qtpu_gpu_pre": preptime,
        "qtpu_gpu_comp": compile_time,
        "qtpu_gpu_run": runtime,
        "qtpu_gpu_post": exec_time,
        "num_instances": num_circuits,
    }


def run_cutensor(circuit: QuantumCircuit):
    from cuquantum import Network, CircuitToEinsum
    import cupy as cp

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

    return float(np.real(result)), {
        "cutensor_compile": compile_time,
        "cutensor_exec": exec_time,
        "cutensor_cost": info.opt_cost,
    }


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
