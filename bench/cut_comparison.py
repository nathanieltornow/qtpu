from time import perf_counter

from qiskit.circuit import QuantumCircuit

import qvm
from qvm.virtual_gates import VirtualBinaryGate, VirtualSWAP
from bench._util import append_to_csv_file


def bench_cut(
    bench_name: str,
    circuits: list[QuantumCircuit],
    csv_name: str | None = None,
    qpu_size: int = 5,
) -> None:
    if csv_name is None:
        csv_name = f"bench_cut_{bench_name}_{qpu_size}.csv"
    fields = [
        "num_qubits",
        "gate_overhead",
        "wire_overhead",
        "optimal_overhead",
        "gate_cut_time",
        "wire_cut_time",
        "optimal_cut_time",
        "gate_num_fragments",
        "wire_num_fragments",
        "optimal_num_fragments",
    ]
    for circuit in circuits:
        now = perf_counter()
        wire_cut_circ = find_cut(circuit, 100, 0, qpu_size)
        wire_cut_time = perf_counter() - now
        now = perf_counter()
        gate_cut_circ = find_cut(circuit, 0, 100, qpu_size)
        gate_cut_time = perf_counter() - now
        now = perf_counter()
        optimal_cut_circ = find_cut(circuit, 100, 100, qpu_size)
        optimal_cut_time = perf_counter() - now

        append_to_csv_file(
            csv_name,
            {
                fields[0]: circuit.num_qubits,
                fields[1]: overhead(gate_cut_circ),
                fields[2]: overhead(wire_cut_circ),
                fields[3]: overhead(optimal_cut_circ),
                fields[4]: gate_cut_time,
                fields[5]: wire_cut_time,
                fields[6]: optimal_cut_time,
                fields[7]: len(gate_cut_circ.qregs),
                fields[8]: len(wire_cut_circ.qregs),
                fields[9]: len(optimal_cut_circ.qregs),
            },
        )


def overhead(circuit: QuantumCircuit) -> int:
    num_vgates = sum(
        1 for instr in circuit if isinstance(instr.operation, VirtualBinaryGate)
    )
    num_vswaps = sum(1 for instr in circuit if isinstance(instr.operation, VirtualSWAP))
    return pow(6, num_vgates) * pow(4, num_vswaps)


def find_cut(
    circuit: QuantumCircuit, num_wire_cuts: int, num_gate_cuts: int, qpu_size: int
) -> QuantumCircuit:
    for frags in range(2, 6):
        try:
            circuit = qvm.cut(
                circuit=circuit,
                technique="optimal",
                max_wire_cuts=num_wire_cuts,
                max_gate_cuts=num_gate_cuts,
                num_fragments=frags,
                max_fragment_size=qpu_size,
            )
            print(f"Found cut with {frags} fragments.")
            return qvm.fragment_circuit(circuit)
        except ValueError:
            continue
    return circuit


if __name__ == "__main__":
    from circuits.qaoa import qaoa
    from circuits.qft import qft
    from circuits.two_local import two_local
    from circuits.vqe import vqe

    bench_cut(
        "qft",
        [
            qft(6, 0),
            qft(8, 0),
            # two_local(12, 3, "linear"),
            # two_local(14, 3, "linear"),
            # two_local(16, 3, "linear"),
            # two_local(12, 3),
            # two_local(14, 3),
            # vqe(6, 3),
            # vqe(8, 3),
            # vqe(10, 3),
            # vqe(12, 3),
            # two_local(15, 3, "linear"),
        ],
    )
