from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
import cotengra as ctg

from qtpu.transforms import remove_operations_by_name
from qtpu.heinsum import HEinsum


def estimate_runtime(circuits: list[QuantumCircuit]) -> float:
    """Estimate the total runtime for executing the given circuits on the specified backend.

    Parameters:
        circuits (list[QuantumCircuit]): The list of quantum circuits to be executed.
        shots (int): The number of shots for each circuit execution.

    Returns:
        float: The estimated total runtime in seconds for executing all circuits.
    """

    shots = 1000
    backend = FakeMarrakesh()
    circuits = [
        remove_operations_by_name(c, {"qpd_measure", "iswitch"}, inplace=False)
        for c in circuits
    ]
    # --- Transpile + schedule ---
    scheduled_circuits = transpile(
        circuits=circuits,
        backend=backend,
        optimization_level=3,
        scheduling_method="asap",
    )

    def _runtime(circuit: QuantumCircuit) -> float:
        # --- Print scheduled circuit ---
        # --- Extract scheduled duration in seconds ---
        dt = backend.configuration().dt
        t_sched = circuit.duration * dt

        # --- Reset / initialization time (not part of schedule) ---
        t_init = 100e-6

        # --- Classical control latency (not in schedule) ---
        t_latency = 20.0 * 1e-6

        # --- Per-shot runtime ---
        t_per_shot = t_init + t_sched + t_latency

        return t_per_shot * shots

    return sum(_runtime(circ) for circ in scheduled_circuits)


def analyse_circuit(circuit: QuantumCircuit) -> dict[str, float]:
    return {
        "num_1q_gates": sum(1 for instr in circuit if instr.operation.num_qubits == 1),
        "num_2q_gates": sum(1 for instr in circuit if instr.operation.num_qubits == 2),
        "depth": circuit.depth(),
        "width": circuit.num_qubits,
    }


def circuit_error(circuit: QuantumCircuit) -> float:
    error = 0.0
    for instr in circuit.data:
        if instr.operation.num_qubits == 2:
            error += 0.01  # Example error rate for CNOT
        elif instr.operation.num_qubits == 1:
            error += 0.001  # Example error rate for single-qubit gates
    return error


def analyze_heinsum(hybrid_tn: HEinsum) -> dict[str, float]:
    subcircuits = [qt.circuit for qt in hybrid_tn.quantum_tensors]

    # Compute sampling overhead from the tensor network
    # This is the product of dimensions at cut boundaries

    opt = ctg.HyperOptimizer()

    inputs, outputs = ctg.utils.eq_to_inputs_output(hybrid_tn.einsum_expr)
    tree: ctg.ContractionTree = opt.search(inputs, outputs, hybrid_tn.size_dict)
    c_cost = tree.contraction_cost()

    return {
        "num_qtensors": len(subcircuits),
        "qtensor_depths": [subcirc.depth() for subcirc in subcircuits],
        "qtensor_widths": [subcirc.num_qubits for subcirc in subcircuits],
        "qtensor_errors": [circuit_error(subcirc) for subcirc in subcircuits],
        "qtensor_num_2q_gates": [
            sum(1 for instr in subcirc if instr.operation.num_qubits == 2)
            for subcirc in subcircuits
        ],
        "num_ctensors": len(hybrid_tn.classical_tensors),
        "c_cost": c_cost,
    }


# def create_baseline_data(circuit: QuantumCircuit) -> dict[str, float]:
#     return {
#         "num_1q_gates": sum(1 for instr in circuit if instr.operation.num_qubits == 1),
#         "num_2q_gates": sum(1 for instr in circuit if instr.operation.num_qubits == 2),
#         "depth": circuit.depth(),
#         "width": circuit.num_qubits,
#         "error": circuit_error(circuit),
#     }


# import pandas as pd
# from mqt.bench import get_benchmark_indep


# def base_line_df(df: pd.DataFrame) -> pd.DataFrame:
#     for index, row in df.iterrows():
#         circuit_size = row["config.circuit_size"]
#         bench_name = row["config.bench"]
#         circuit = get_benchmark_indep(bench_name, circuit_size)
#         baseline_data = create_baseline_data(circuit)
#         for key, value in baseline_data.items():
#             df.at[index, f"baseline.{key}"] = value
#     return df
