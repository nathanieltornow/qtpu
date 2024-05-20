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


def generate_benchmark(name: str, num_qubits: int) -> QuantumCircuit:
    if name in mqt_benchmarks:
        circuit = get_benchmark(name, 1, num_qubits)
        circuit.remove_final_measurements()
        _measure_all(circuit)
        return circuit

    current_path = Path(__file__).parent

    if name in qvm_benchmarks:
        return QuantumCircuit.from_qasm_file(
            current_path / "circuits" / f"{name}" / f"{num_qubits}.qasm"
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
