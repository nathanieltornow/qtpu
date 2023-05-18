from multiprocessing import Pool
from time import perf_counter

from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend, BackendV2, BackendV2Converter
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile

import qvm
from qvm.virtual_gates import VirtualBinaryGate
from qvm.util import virtualize_between_qubits

from csv_util import append_to_csv_file
from fidelity import calculate_total_variation_distance, calcultate_fidelity


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

        # coupling_map = backend.coupling_map
        # t_circuit = transpile(
        #     circuit, backend=backend, optimization_level=optimization_level
        # )
        # init_layout = initial_layout_from_transpiled_circuit(circuit, t_circuit)

        now = perf_counter()
        v_circuit, _ = virtualize_between_qubits(circuit, qubit1=circuit.qubits[0], qubit2=circuit.qubits[-1])
        # v_circuit = qvm.vroute(
        #     circuit=circuit,
        #     technique=vroute_technique,
        #     coupling_map=coupling_map,
        #     initial_layout=init_layout,
        #     max_gate_cuts=max_overhead // 6,
        # )
        cut_time = perf_counter() - now
        virtualizer = qvm.OneFragmentGateVirtualizer(v_circuit)
        frag, frag_circuit = list(virtualizer.fragments().items())[0]
        args = virtualizer.instantiate()[frag]
        frag_circuit = transpile(frag_circuit, backend=backend, optimization_level=optimization_level)

        circuits_to_run = [qvm.insert_placeholders(frag_circuit, arg) for arg in args]
        now = perf_counter()
        print("running...")
        counts = backend.run(circuits_to_run, shots=num_shots).result().get_counts()
        assert isinstance(counts, list)
        distrs = [
            qvm.QuasiDistr.from_counts(count, shots=num_shots) for count in counts
        ]
        run_time = perf_counter() - now
        print("knitting...")
        with Pool() as pool:
            now = perf_counter()
            res_distr = virtualizer.knit({frag: distrs}, pool=pool)
            knit_time = perf_counter() - now
            
        t_circuit = transpile(circuit, backend=backend, optimization_level=optimization_level)
        base_counts = backend.run(t_circuit, shots=num_shots).result().get_counts()
        base_distr = qvm.QuasiDistr.from_counts(base_counts, shots=num_shots)

        from qvm.quasi_distr import QuasiDistr

        fid = calcultate_fidelity(circuit, QuasiDistr.from_counts(res_distr.to_counts(num_shots)))
        base_fid = calcultate_fidelity(circuit, base_distr)

        append_to_csv_file(
            csv_name,
            {
                fields[0]: circuit.num_qubits,
                fields[1]: base_fid,
                fields[2]: fid,
                fields[3]: num_cnots(t_circuit),
                fields[4]: num_cnots(frag_circuit),
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
    from circuits.ae import ae
    from circuits.qaoa import qaoa
    from circuits.vqe import vqe
    from circuits.two_local import two_local
    from qiskit.providers.fake_provider import FakeOslo, FakeMontrealV2
    from qiskit_aer import AerSimulator
    from adaptive_noisemodel import get_noisemodel

    circuits = [two_local(8, 1), two_local(12, 1), two_local(16, 1)]
    backend = FakeMontrealV2()
    bench_virtual_routing("2local", circuits, backend, max_overhead=300)
