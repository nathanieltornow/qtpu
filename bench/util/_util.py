from dataclasses import dataclass
import json
import sys
import os
import csv

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers import BackendV2
from qiskit.quantum_info import hellinger_fidelity

from qvm.virtual_gates import VirtualBinaryGate, VirtualMove, VirtualGateEndpoint
from qvm.qvm_runner import QVMBackendRunner
from qvm.virtual_circuit import VirtualCircuit


def transpile_virtualizer(
    virtualizer: VirtualCircuit, backend: BackendV2, optimization_level: int = 3
) -> None:
    fragment_circs = list(virtualizer.fragment_circuits.items())
    for frag, circ in fragment_circs:
        t_circ = transpile(circ, backend, optimization_level=optimization_level)
        virtualizer.replace_fragment_circuit(frag, t_circ)


def get_num_cnots(circuit: QuantumCircuit) -> int:
    return sum(1 for instr in circuit if instr.operation.name == "cx")


def overhead(circuit: QuantumCircuit) -> int:
    num_vgates = sum(
        1 for instr in circuit if isinstance(instr.operation, VirtualBinaryGate)
    )
    num_wire_cuts = sum(
        1 for instr in circuit if isinstance(instr.operation, VirtualMove)
    )
    return 4**num_vgates * 6**num_wire_cuts


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


def total_variation_distance(p: dict[int, float], q: dict[int, float]):
    events = set(p.keys()).union(set(q.keys()))
    tv_distance = 0.0
    for event in events:
        p_prob = p.get(event, 0.0)
        q_prob = q.get(event, 0.0)
        tv_distance += 0.5 * abs(p_prob - q_prob)
    return tv_distance


def compute_fidelity(
    circuit: QuantumCircuit, noisy_distr: dict[int, float], runner: QVMBackendRunner
) -> tuple[float, float]:
    job = runner.run(circuit)
    perfect_distr = runner.get_results(job)[0].nearest_probability_distribution()
    tv_fid = 1 - total_variation_distance(perfect_distr, noisy_distr)
    hel_fid = hellinger_fidelity(perfect_distr, noisy_distr)
    return hel_fid, tv_fid
