from time import perf_counter

from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend, BackendV2, BackendV2Converter
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile

import qvm
from qvm.virtual_gates import VirtualBinaryGate

from csv_util import append_to_csv_file


def bench_virtual_routing(
    bench_name: str,
    circuits: list[QuantumCircuit],
    backend: Backend,
    csv_name: str | None = None,
    vroute_technique: str = "perfect",
    num_shots: int = 20000,
    max_overhead: int = 1296,
    optimization_level: int = 3,
) -> None:
    if not isinstance(backend, BackendV2):
        backend = BackendV2Converter(backend)

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

        coupling_map = backend.coupling_map
        t_cricuit = transpile(
            circuit, backend=backend, optimization_level=optimization_level
        )
        init_layout = initial_layout_from_transpiled_circuit(circuit, t_cricuit)

        now = perf_counter()
        virt_circ = qvm.vroute(
            circuit, vroute_technique
        )
        pass


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
