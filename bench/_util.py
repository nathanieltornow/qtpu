import os
import csv

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

from qvm.quasi_distr import QuasiDistr
from qvm.util import circuit_to_qcg
from qvm.virtual_gates import VirtualBinaryGate


def average_degree(cricuit: QuantumCircuit) -> float:
    qcg = circuit_to_qcg(cricuit, use_qubit_idx=True)
    return sum([d for _, d in qcg.degree(weight="weight")]) / qcg.number_of_nodes()


def num_cnots(circuit: QuantumCircuit) -> int:
    return sum(1 for instr in circuit if instr.operation.name == "cx")


def overhead(circuit: QuantumCircuit) -> int:
    num_vgates = sum(
        1 for instr in circuit if isinstance(instr.operation, VirtualBinaryGate)
    )
    return pow(6, num_vgates)


def initial_layout_from_transpiled_circuit(
    circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit
) -> list[int]:
    if transpiled_circuit._layout is None:
        raise ValueError("Circuit has no layout.")
    init_layout = [0] * circuit.num_qubits
    qubit_to_index = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    for p, q in transpiled_circuit._layout.initial_layout.get_physical_bits().items():
        if q in qubit_to_index:
            init_layout[qubit_to_index[q]] = p
    return init_layout


def total_variation_distance(p: QuasiDistr, q: QuasiDistr):
    events = set(p.keys()).union(set(q.keys()))
    tv_distance = 0.0
    for event in events:
        p_prob = p.get(event, 0.0)
        q_prob = q.get(event, 0.0)
        tv_distance += 0.5 * abs(p_prob - q_prob)
    return tv_distance


def calculate_total_variation_distance(
    circuit: QuantumCircuit, noisy_distr: QuasiDistr
) -> float:
    return total_variation_distance(perfect_distr(circuit), noisy_distr)


def perfect_distr(circuit: QuantumCircuit):
    sim = AerSimulator(method="statevector")
    counts = sim.run(circuit, shots=20000).result().get_counts()
    return QuasiDistr.from_counts(counts, shots=20000)


def calcultate_fidelity(circuit: QuantumCircuit, noisy_distr: QuasiDistr) -> float:
    return hellinger_fidelity(perfect_distr(circuit), noisy_distr)


def append_to_csv_file(filepath: str, data: dict[str, int | float]) -> None:
    if not os.path.exists(filepath):
        # os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as csv_file:
            csv.DictWriter(csv_file, fieldnames=data.keys()).writeheader()
            csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)
        return

    with open(filepath, "a") as csv_file:
        csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)
