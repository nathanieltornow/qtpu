import os
import csv
import numpy as np
from dataclasses import dataclass, asdict

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators import Pauli
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator
from qiskit.providers import BackendV2
from qiskit.compiler import transpile

from qvm import VirtualCircuit, build_dummy_tensornetwork
from qvm.virtual_gates import VirtualMove


@dataclass
class VirtualCircuitInfo:
    n_fragments: int
    fragment_size: int
    n_instantiations: int
    n_max_instantiations: int
    n_virtual_gates: int
    knit_cost: int
    naive_knit_cost: int

    depth: int = np.nan
    n_binary_gates: int = np.nan
    eps: float = np.nan

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def get_virtual_circuit_info(
    vc: VirtualCircuit,
    backend: BackendV2 | None = None,
    opt_level: int = 3,
) -> VirtualCircuitInfo:
    n_frags = len(vc.fragments)
    frag_size = max(len(frag) for frag in vc.fragments)
    n_instantiations = sum(vc.num_instantiations().values())
    n_max_instantiations = max(vc.num_instantiations().values())
    n_virtual_gates = len(vc.virtual_gates)

    tn = build_dummy_tensornetwork(vc)
    knit_cost = tn.contraction_cost(optimize="auto")

    vgate_overhead = np.prod(
        [8 if isinstance(vgate, VirtualMove) else 6 for vgate in vc.virtual_gates]
    )
    naive_knit_cost = (vgate_overhead - 1) * (n_frags - 1)

    depth = np.nan
    n_binary_gates = np.nan
    eps = np.nan

    if backend is not None:
        frag_circs = [
            transpile(frag, backend=backend, optimization_level=opt_level)
            for frag in vc.fragment_circuits.values()
        ]
        depth = max(circ.depth() for circ in frag_circs)
        n_binary_gates = max(
            sum(1 for instr in circ if len(instr.qubtis) == 2) for circ in frag_circs
        )
        eps = 1.0

    info = VirtualCircuitInfo(
        n_fragments=n_frags,
        fragment_size=frag_size,
        n_instantiations=n_instantiations,
        n_max_instantiations=n_max_instantiations,
        n_virtual_gates=n_virtual_gates,
        knit_cost=knit_cost,
        naive_knit_cost=naive_knit_cost,
        depth=depth,
        n_binary_gates=n_binary_gates,
        eps=eps,
    )

    return info


def append_to_csv(file_path: str, data: dict) -> None:
    if not os.path.exists(file_path):
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
