from multiprocessing import Pool
from time import perf_counter

from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend, BackendV2, BackendV2Converter
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile

import qvm
from qvm.virtual_gates import VirtualBinaryGate

from csv_util import append_to_csv_file
from fidelity import calcultate_fidelity


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
        "run_time",
    ]

    for circuit in circuits:
        if circuit.num_qubits > backend.num_qubits:
            raise ValueError("Circuit has more qubits than backend.")

        coupling_map = backend.coupling_map
        t_circuit = transpile(
            circuit, backend=backend, optimization_level=optimization_level
        )
        init_layout = initial_layout_from_transpiled_circuit(circuit, t_circuit)

        now = perf_counter()
        v_circuit = qvm.vroute(
            circuit=circuit,
            technique=vroute_technique,
            coupling_map=coupling_map,
            initial_layout=init_layout,
            max_gate_cuts=max_overhead // 6,
        )
        cut_time = perf_counter() - now
        virtualizer = qvm.OneFragmentGateVirtualizer(v_circuit)
        frag, circuit = list(virtualizer.fragments().items())[0]
        args = virtualizer.instantiate()[frag]
        circuits_to_run = [qvm.insert_placeholders(circuit, arg) for arg in args]
        now = perf_counter()
        counts = backend.run(circuits_to_run, shots=num_shots).result().get_counts()
        assert isinstance(counts, list)
        distrs = [
            qvm.QuasiDistr.from_counts(count, shots=num_shots) for count in counts
        ]
        run_time = perf_counter() - now

        with Pool() as pool:
            now = perf_counter()
            res_distr = virtualizer.knit({frag: distrs}, pool=pool)
            knit_time = perf_counter() - now

        base_counts = backend.run(t_circuit, shots=num_shots).result().get_counts()
        base_distr = qvm.QuasiDistr.from_counts(base_counts, shots=num_shots)

        fid = calcultate_fidelity(circuit, res_distr)
        base_fid = calcultate_fidelity(circuit, base_distr)

        append_to_csv_file(
            csv_name,
            {
                fields[0]: circuit.num_qubits,
                fields[1]: base_fid,
                fields[2]: fid,
                fields[3]: num_cnots(t_circuit),
                fields[4]: num_cnots(v_circuit),
                fields[5]: overhead(v_circuit),
                fields[6]: cut_time,
                fields[7]: knit_time,
                fields[8]: run_time,
            },
        )


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


if __name__ == "__main__":
    from circuits.vqe import vqe
    from qiskit.providers.fake_provider import FakeOslo

    circuits = [vqe(4, 1), vqe(5, 1)]
    backend = FakeOslo()
    bench_virtual_routing("vqe", circuits, backend)
