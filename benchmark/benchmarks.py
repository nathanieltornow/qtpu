from pathlib import Path

from qiskit.circuit import QuantumCircuit, ClassicalRegister

from mqt.bench import get_benchmark

mqt_benchmarks = [
    "ae",
    "dj",
    "grover-noancilla",
    "grover-v-chain",
    "ghz",
    "graphstate",
    "portfolioqaoa",
    "portfoliovqe",
    "qaoa",
    "qft",
    "qftentangled",
    "qnn",
    "qpeexact",
    "qpeinexact",
    "qwalk-noancilla",
    "qwalk-v-chain",
    "random",
    "realamprandom",
    "su2random",
    "twolocalrandom",
    "vqe",
    "wstate",
    "shor",
    "pricingcall",
    "pricingput",
    "groundstate",
    "routing",
    "tsp",
]

qvm_benchmarks = [
    "qaoa_r4",
    "hamsim_3",
    "vqe_3",
    "qaoa_r3",
    "qaoa_ba3",
    "qsvm",
    "wstate",
    "twolocal_3",
    "qaoa_ba4",
    "vqe_2",
    "twolocal_1",
    "qaoa_r2",
    "adder",
    "hamsim_2",
    "qaoa_b",
    "ghz",
    "qaoa_ba1",
    "twolocal_2",
    "vqe_1",
    "bv",
    "hamsim_1",
    "qaoa_ba2",
]

pretty_names = {
    "hamsim_1": "HS-1",
    "hamsim_2": "HS-2",
    "hamsim_3": "HS-3",
    "qsvm": "QSVM",
    "qaoa_b": "QAOA-B",
    "qaoa_ba1": "QAOA-BA1",
    "qaoa_ba2": "QAOA-BA2",
    "qaoa_ba3": "QAOA-BA3",
    "qaoa_ba4": "QAOA-BA4",
    "qaoa_r2": "QAOA-R2",
    "qaoa_r3": "QAOA-R3",
    "qaoa_r4": "QAOA-R4",
    "qft": "QFT",
    "twolocal_1": "TL-1",
    "twolocal_2": "TL-2",
    "twolocal_3": "TL-3",
    "vqe_1": "VQE-1",
    "vqe_2": "VQE-2",
    "vqe_3": "VQE-3",
    "wstate": "W-state",
    "ghz": "GHZ",
}


def generate_benchmark(name: str, num_qubits: int) -> QuantumCircuit:
    if name in mqt_benchmarks:
        circuit = get_benchmark(name, 1, num_qubits)
        return _remove_barrier(circuit)

    current_path = Path(__file__).parent

    if name in qvm_benchmarks:
        return _remove_barrier(
            QuantumCircuit.from_qasm_file(
                current_path / "circuits" / f"{name}" / f"{num_qubits}.qasm"
            )
        )

    raise ValueError(f"Unknown benchmark name: {name}")


def generate_benchmarks_range(
    name: str, min_qubits: int, max_qubits: int
) -> list[QuantumCircuit]:
    benches = []
    for num_qubits in range(min_qubits, max_qubits + 1):
        try:
            benches.append(generate_benchmark(name, num_qubits))
        except Exception:
            continue
    return benches


def _measure_all(circuit: QuantumCircuit) -> None:
    assert circuit.num_clbits == 0
    creg = ClassicalRegister(circuit.num_qubits)
    circuit.add_register(creg)
    circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))


def _remove_barrier(circuit: QuantumCircuit) -> QuantumCircuit:
    new_circ = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instr in circuit:
        if instr.operation.name == "barrier":
            continue
        new_circ.append(instr, instr.qubits, instr.clbits)
    return new_circ