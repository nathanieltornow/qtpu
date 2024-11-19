from typing import Any, Sequence
import os

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import optuna
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators import Pauli
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator, AerSimulator
from qiskit.providers import BackendV2
from qiskit.circuit.library import SwapGate, CXGate
from qiskit.primitives import BaseSampler, SamplerResult, PrimitiveJob
from qiskit.result import QuasiDistribution
from qiskit.compiler import transpile

from qiskit_addon_cutting.utils.iteration import strict_zip
from qiskit_addon_cutting.qpd import TwoQubitQPDGate
import cotengra as ctg

from qtpu.tensor import HybridTensorNetwork
from qtpu.circuit import cuts_to_moves, circuit_to_hybrid_tn


def concat_data(file_path: str, data: dict):
    if os.path.exists(file_path):
        # Read the existing JSON file into a DataFrame
        df = pd.read_json(file_path)
    else:
        # Create an empty DataFrame if the file doesn't exist
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_json(file_path, orient="records", indent=4)


def get_info(circuit: QuantumCircuit, backend: BackendV2 | None = None) -> dict:
    circuit = cuts_to_moves(circuit)
    qpds = [
        instr.operation.basis
        for instr in circuit
        if isinstance(instr.operation, TwoQubitQPDGate)
    ]
    htn = circuit_to_hybrid_tn(circuit)

    sub_circuits = [
        transpile(qt._circuit, backend=backend, optimization_level=3)
        for qt in htn.quantum_tensors
    ]
    circuit = replace_qpd_gates(circuit)
    circuit = transpile(circuit, backend=backend, optimization_level=3)

    return {
        "qtpu_cost_log10": np.log10(
            htn.to_tensor_network().contraction_cost(optimize="auto") + 1
        ),
        "ckt_cost_log10": np.sum([np.log10(len(qpd.coeffs)) for qpd in qpds])
        + np.log10(len(qpds) + len(htn.quantum_tensors)),
        "num_qpds": len(qpds),
        "num_subcircuits": len(htn.quantum_tensors),
        "num_instances": np.sum([qt.ind_tensor.size for qt in htn.quantum_tensors]),
        "num_2q": max([circuit.num_nonlocal_gates() for circuit in sub_circuits]),
        "max_qubits": max([circuit.num_qubits for circuit in sub_circuits]),
        "depth": max([circuit.depth() for circuit in sub_circuits]),
        "esp": min([esp(circuit) for circuit in sub_circuits]),
        "base_qubits": circuit.num_qubits,
        "base_depth": circuit.depth(),
        "base_num_2q": circuit.num_nonlocal_gates(),
        "base_esp": esp(circuit),
    }


def replace_qpd_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    circ = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instr in circuit:
        if isinstance(instr.operation, TwoQubitQPDGate):
            circ.append(CXGate(), instr.qubits)
        else:
            circ.append(instr, instr.qubits)
    return circ


def ckt_cost(circuit: QuantumCircuit, num_samples: int = np.inf) -> int:
    circuit = cuts_to_moves(circuit)
    qpds = [
        instr.operation.basis
        for instr in circuit
        if isinstance(instr.operation, TwoQubitQPDGate)
    ]
    return min(np.prod([len(qpd.coeffs) for qpd in qpds]), num_samples)


def qtpu_cost(circuit: QuantumCircuit, tolerance: float = 0.0) -> int:
    circuit = cuts_to_moves(circuit)
    htn = circuit_to_hybrid_tn(circuit)
    htn.simplify(tolerance)
    return htn.to_tensor_network().contraction_cost(optimize="auto")


def get_hybrid_tn_info(
    hybrid_tn: HybridTensorNetwork, backend: BackendV2 | None = None
) -> dict:

    # if backend is None:
    #     backend = AerSimulator()

    circuits = [
        transpile(qt.circuit, backend=backend, optimization_level=3)
        for qt in hybrid_tn.quantum_tensors
    ]

    return {
        "contract_cost": round(contraction_cost_log10(hybrid_tn), 3),
        "bruteforce_cost": round(bruteforce_cost_log10(hybrid_tn), 3),
        "num_qpds": len(hybrid_tn.classical_tensors),
        "num_subcircuits": len(circuits),
        "num_instances": sum([np.prod(qt.shape) for qt in hybrid_tn.quantum_tensors]),
        "esp": min([esp(circuit) for circuit in circuits]),
        "depth": max([circuit.depth() for circuit in circuits]),
        "max_qubits": max([circuit.num_qubits for circuit in circuits]),
        "swap_count": max([circuit.count_ops().get("swap", 0) for circuit in circuits]),
        "cx_count": max([circuit.count_ops().get("cx", 0) for circuit in circuits]),
        "2q_count": max([circuit.num_nonlocal_gates() for circuit in circuits]),
    }


def get_base_info(circuit: QuantumCircuit, backend: BackendV2 | None = None) -> dict:
    # if backend is None:
    #     backend = AerSimulator()

    circuit = transpile(circuit, backend=backend, optimization_level=3)

    return {
        "base_qubits": circuit.num_qubits,
        "base_depth": circuit.depth(),
        "base_esp": esp(circuit),
        "base_swap_count": circuit.count_ops().get("swap", 0),
        "base_cx_count": circuit.count_ops().get("cx", 0),
        "base_2q_count": circuit.num_nonlocal_gates(),
    }


def contraction_cost_log10(hybrid_tn: HybridTensorNetwork) -> int:
    if hybrid_tn.size_dict() == {}:
        return 0
    opt = ctg.HyperOptimizer()

    return opt.search(
        hybrid_tn.inputs(), hybrid_tn.output(), hybrid_tn.size_dict()
    ).contraction_cost(log=10)


def bruteforce_cost_log10(hybrid_tn: HybridTensorNetwork) -> int:
    return np.sum(
        [np.log10(tens.shape[0]) for tens in hybrid_tn.classical_tensors]
    ) + np.log10(len(hybrid_tn.classical_tensors) + len(hybrid_tn.quantum_tensors) - 1)


def append_to_csv(file_path: str, data: dict) -> None:
    if not os.path.exists(file_path):
        if os.path.dirname(file_path) != "":
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writeheader()
    with open(file_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(data)


def get_perfect_z_expval(circuit: QuantumCircuit) -> float:
    circuit = circuit.remove_final_measurements(inplace=False)
    op = Pauli("Z" * circuit.num_qubits)
    sv: Statevector = StatevectorSimulator().run(circuit).result().get_statevector()
    return sv.expectation_value(op)


def esp(circuit: QuantumCircuit) -> float:
    fid = 1.0
    for instr in circuit:
        op = instr.operation

        if op.name == "barrier":
            continue

        if op.name == "measure":
            fid *= 1 - 1e-3

        elif op.num_qubits == 1:
            fid *= 1 - 1e-4

        elif op.num_qubits == 2:
            fid *= 1 - 1e-3

        else:
            raise ValueError(f"Unsupported operation: {op}")

    return round(fid, 3)


def postprocess_barplot(ax: plt.Axes) -> None:
    hatches = ["/", "\\", "//", "\\\\", "x", ".", ",", "*"]
    num_xticks = len(ax.get_xticks())
    num_bars = len(ax.get_legend_handles_labels()[0])
    patch_idx_to_hatch_idx = np.arange(num_bars).repeat(num_xticks)
    for i, patch in enumerate(ax.patches):
        patch.set_hatch(hatches[patch_idx_to_hatch_idx[i] % len(hatches)])


class DummySampler(BaseSampler):

    def _call(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: Sequence[Sequence[float]],
        **ignored_run_options,
    ) -> SamplerResult:
        metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]
        bound_circuits = [
            circuit if len(value) == 0 else circuit.assign_parameters(value)
            for circuit, value in strict_zip(circuits, parameter_values)
        ]
        probabilities = [{0: 1.0} for qc in bound_circuits]
        quasis = [QuasiDistribution(p) for p in probabilities]
        return SamplerResult(quasis, metadata)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ):
        job = PrimitiveJob(self._call, circuits, parameter_values, **run_options)
        # The public submit method was removed in Qiskit 1.0
        (job.submit if hasattr(job, "submit") else job._submit)()
        return job
