from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.compiler import transpile

from qvm.qvm_runner import QVMBackendRunner
from qvm.run import run_virtualizer
from qvm.virtualizer import Virtualizer


from _util import (
    compute_fidelity,
    get_num_cnots,
    append_to_csv_file,
    transpile_virtualizer,
    CircuitProperties,
)


def run_experiment(
    csv_name: str,
    original_circuit: QuantumCircuit,
    virt: Virtualizer,
    runner: QVMBackendRunner,
    backend: BackendV2,
    base_backend: BackendV2 | None = None,
) -> None:
    if base_backend is None:
        base_backend = backend

    assert (
        base_backend.num_qubits >= original_circuit.num_qubits
        and backend.num_qubits >= original_circuit.num_qubits
    )

    result, timing = run_virtualizer(virt, runner, backend)
    num_vgates = len(virt._vgate_instrs)

    h_fid, tv_fid = 0.0, 0.0
    if original_circuit.num_qubits < 25:
        h_fid, tv_fid = compute_fidelity(original_circuit, result, runner)

    h_fid_base, tv_fid_base = 0.0, 0.0
    t_circuit = transpile(original_circuit, base_backend, optimization_level=3)
    job_id = runner.run([t_circuit], base_backend)
    noisy_base_res = runner.get_results(job_id)[0].nearest_probability_distribution()

    if original_circuit.num_qubits < 25:
        h_fid_base, tv_fid_base = compute_fidelity(
            original_circuit, noisy_base_res, runner
        )

    append_to_csv_file(
        csv_name,
        {
            "num_qubits": original_circuit.num_qubits,
            "h_fid_baseline": h_fid_base,
            "tv_fid_baseline": tv_fid_base,
            "h_fid": h_fid,
            "tv_fid": tv_fid,
            "num_vgates": num_vgates,
            "run_time": timing.run_time,
            "knit_time": timing.knit_time,
        },
    )


def get_circuit_properties(
    original_circuit: QuantumCircuit,
    virtualizer: Virtualizer,
    backend: BackendV2,
    base_backend: BackendV2 | None = None,
) -> CircuitProperties:
    transpile_virtualizer(virtualizer, backend)

    if base_backend is None:
        base_backend = backend

    num_qubits = original_circuit.num_qubits

    frag_circs = virtualizer.fragment_circuits.values()

    num_cnots = max(get_num_cnots(circ) for circ in frag_circs)
    depth = max(circ.depth() for circ in frag_circs)

    t_circ = transpile(original_circuit, backend=base_backend, optimization_level=3)
    num_cnots_base = get_num_cnots(t_circ)
    depth_base = t_circ.depth()

    return CircuitProperties(
        num_qubits=num_qubits,
        num_cnots=num_cnots,
        num_cnots_base=num_cnots_base,
        depth=depth,
        depth_base=depth_base,
        num_vgates=len(virtualizer._vgate_instrs),
        num_fragments=len(virtualizer.fragment_circuits),
    )
