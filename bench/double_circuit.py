from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend, BackendV2, BackendV2Converter

import qvm

def bench_double_circuit(
    bench_name: str,
    circuits: list[QuantumCircuit],
    backend: Backend,
    csv_name: str | None = None,
    vroute_technique: str = "perfect",
    num_shots: int = 20000,
    max_overhead: int = 1296,
    optimization_level: int = 3,
):  
    if not isinstance(backend, BackendV2):
        backend = BackendV2Converter(backend)

    if csv_name is None:
        csv_name = f"bench_vroute_{bench_name}.csv"
    fields = [
        "num_qubits",
        "base_fidelity",
        "virtual_fidelity",
        "overhead",
        "cut_time",
        "knit_time",
        "run_time",
    ]

    for circuit in circuits:
        virt_circuit = qvm.cut(circuit, technique="gate_bisection", num_fragments=2)

        # get a virtualizer
        virt = qvm.TwoFragmentGateVirtualizer(virt_circuit)
        frag_circs = virt.fragments()

        simulator = AerSimulator()
        results = {}
        for fragment, args in virt.instantiate().items():
            circuits = [qvm.insert_placeholders(frag_circs[fragment], arg) for arg in args]
            counts = simulator.run(circuits, shots=SHOTS).result().get_counts()
            counts = [counts] if isinstance(counts, dict) else counts
            results[fragment] = [qvm.QuasiDistr.from_counts(count) for count in counts]

        with Pool() as pool:
            res_distr = virt.knit(results, pool=pool)