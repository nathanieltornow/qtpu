from time import perf_counter

from qiskit.circuit import QuantumCircuit
from qiskit.providers.fake_provider import FakeBackendV2

import qvm
from qvm.virtual_gates import VirtualBinaryGate

from .csv_util import append_to_csv_file


def bench_virtual_routing(
    bench_name: str,
    circuits: list[QuantumCircuit],
    backend: FakeBackendV2,
    csv_name: str | None = None,
    vroute_technique: str = "perfect",
    num_shots: int = 20000,
    max_overhead: int = 1296,
) -> None:
    assert isinstance(backend, FakeBackendV2)

    if csv_name is None:
        csv_name = f"bench_vroute_{bench_name}.csv"
    fields = [
        "num_qubits",
        "base_fidelity",
        "vroute_fidelity",
        "base_num_cx",
        "vroute_num_cx",
        "overhead",
        "cut_time",
        "knit_time",
    ]

    for circuit in circuits:
        if circuit.num_qubits > backend.num_qubits:
            raise ValueError("Circuit has more qubits than backend.")

        now = perf_counter()
        virt_circ = qvm.vr
        pass


def num_cnots(circuit: QuantumCircuit) -> int:
    return sum(1 for instr in circuit if instr.operation.name == "cx")


def overhead(circuit: QuantumCircuit) -> int:
    num_vgates = sum(
        1 for instr in circuit if isinstance(instr.operation, VirtualBinaryGate)
    )
    return pow(6, num_vgates)
