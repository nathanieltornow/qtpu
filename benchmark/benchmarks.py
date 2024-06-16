from pathlib import Path

import numpy as np

import networkx as nx
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import TwoLocal

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


def qaoa(graph: nx.Graph) -> QuantumCircuit:
    # edges = sorted(graph.edges(data=False), key=lambda x: (x[0] + 1) * (x[1] + 1))
    edges = graph.edges(data=False)
    circuit = QuantumCircuit(graph.number_of_nodes())
    for i, j in edges:
        circuit.rzz(2, i, j)
    return circuit


def generate_benchmark(name: str, num_qubits: int) -> QuantumCircuit:
    if name.startswith("qaoa"):
        d = int(name[-1])
        graph = nx.random_regular_graph(d, num_qubits, seed=120)
        return qaoa(graph)

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


def brick_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.rx(np.random.rand() * 2 * np.pi, i)
            circuit.rz(np.random.rand() * 2 * np.pi, i)
        for i in range(0, num_qubits - 1, 2):
            circuit.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            circuit.cx(i, i + 1)
    return circuit


def linear_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.rx(np.random.rand() * 2 * np.pi, i)
            circuit.rz(np.random.rand() * 2 * np.pi, i)
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
    return circuit


def random_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.rx(np.random.rand() * 2 * np.pi, i)
            circuit.rz(np.random.rand() * 2 * np.pi, i)

        # create random tuples of qubits to apply CNOT gates, such that each qubit is in at most once
        qubits = list(range(num_qubits))
        np.random.shuffle(qubits)

        for i in range(0, num_qubits - 1, 2):
            circuit.cx(qubits[i], qubits[i + 1])

    return circuit


def _cluster(num_qubits: int):
    cluster = TwoLocal(num_qubits, "ry", "rzz", reps=1)
    cluster = cluster.assign_parameters(np.random.rand(cluster.num_parameters) * 2 * np.pi)
    return cluster


def cluster_ansatz(cluster_sizes: list[int], depth: int) -> QuantumCircuit:
    circuit = QuantumCircuit(sum(cluster_sizes))
    for _ in range(depth):
        start = 0
        for s in cluster_sizes:
            cluster = _cluster(s)
            circuit.compose(cluster, range(start, start + s), inplace=True)
            start += s
        start = 0
        for s in cluster_sizes[:-1]:
            circuit.cx(start + s - 1, start + s)
            start += s
    return circuit
