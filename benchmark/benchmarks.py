from pathlib import Path
import itertools

import numpy as np

import networkx as nx
from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
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


def ring_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.rx(np.random.rand() * 2 * np.pi, i)
            circuit.rz(np.random.rand() * 2 * np.pi, i)
        for i in range(num_qubits):
            circuit.cx(i, (i + 1) % num_qubits)
    return circuit


def _qaoa(graph: nx.Graph, depth: int) -> QuantumCircuit:
    circuit = QuantumCircuit(graph.number_of_nodes())
    for d in range(depth):
        for i in graph.nodes:
            circuit.rx(np.random.rand() * 2 * np.pi, i)
            circuit.rz(np.random.rand() * 2 * np.pi, i)

        edges = (
            graph.edges(data=False) if d % 2 == 0 else reversed(graph.edges(data=False))
        )

        for i, j in edges:
            circuit.rzz(np.random.rand() * 2 * np.pi, i, j)
    return circuit


def qaoa_regular_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    graph = nx.random_regular_graph(3, num_qubits, seed=123)
    return _qaoa(graph, depth)


def qaoa_powerlaw_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    graph = nx.powerlaw_cluster_graph(num_qubits, 3, 0.0005, seed=123)
    return _qaoa(graph, depth)


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


def _cluster(num_qubits: int, prob: float = 0.8) -> QuantumCircuit:
    cluster = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        cluster.rx(np.random.rand() * 2 * np.pi, i)
        cluster.rz(np.random.rand() * 2 * np.pi, i)
    for q1, q2 in itertools.combinations(range(num_qubits), 2):
        if np.random.rand() < prob:
            cluster.cx(q1, q2)

    circuit = QuantumCircuit(num_qubits)
    circuit.append(cluster.to_gate(), range(num_qubits))
    return circuit


def cluster_ansatz(cluster_sizes: list[int], depth: int) -> QuantumCircuit:
    cluster_regs = [QuantumRegister(s, f"q_{i}") for i, s in enumerate(cluster_sizes)]
    circuit = QuantumCircuit(*cluster_regs)

    for _ in range(depth):
        for qreg in cluster_regs:
            circuit.compose(_cluster(qreg.size), qubits=qreg, inplace=True)

        for i in range(len(cluster_sizes) - 1):
            circuit.cx(cluster_regs[i][-1], cluster_regs[i + 1][0])
    return circuit
