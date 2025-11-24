from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers import BackendV2

from qtpu.transforms import remove_operations_by_name
from qtpu.tensor import HybridTensorNetwork


def estimate_runtime(
    circuits: list[QuantumCircuit], backend: BackendV2, shots: int
) -> float:
    """Estimate the total runtime for executing the given circuits on the specified backend.

    Parameters:
        circuits (list[QuantumCircuit]): The list of quantum circuits to be executed.
        backend (BackendV2): The backend on which the circuits will be executed.
        shots (int): The number of shots for each circuit execution.

    Returns:
        float: The estimated total runtime in seconds for executing all circuits.
    """
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


def estimate_error(circuits: list[QuantumCircuit], backend: BackendV2) -> float:
    """Estimate the total error for executing the given circuits on the specified backend.

    Parameters:
        circuits (list[QuantumCircuit]): The list of quantum circuits to be executed.
        backend (BackendV2): The backend on which the circuits will be executed.

    Returns:
        float: The estimated total error for executing all circuits.
    """
    total_error = 0.0

    for circuit in circuits:
        transpiled_circuit = transpile(
            circuit,
            backend=backend,
            optimization_level=3,
            scheduling_method="asap",
        )

        for instr in transpiled_circuit.data:
            gate_error = backend.gate_error(instr.operation.name)
            if gate_error is not None:
                total_error += gate_error

        # Add measurement errors
        for qubit in range(transpiled_circuit.num_qubits):
            meas_error = backend.properties().readout_error(qubit)
            if meas_error is not None:
                total_error += meas_error

    return total_error


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


def analyze_hybrid_tn(hybrid_tn: HybridTensorNetwork) -> dict[str, float]:
    subcircuits = hybrid_tn.subcircuits
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
        "c_cost": hybrid_tn.to_tensor_network().contraction_cost(optimize="auto"),
    }
