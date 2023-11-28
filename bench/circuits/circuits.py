import os

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit


BENCHMARK_CIRCUITS = [
    # "adder",
    "qaoa_r2",
    # "qaoa_r3",
    # "qaoa_r4",
    # "bv",
    "ghz",
    "hamsim_1",
    "hamsim_2",
    "hamsim_3",
    "qaoa_b",
    "qaoa_ba1",
    "qaoa_ba2",
    # "qaoa_ba3",
    # "qaoa_ba4",
    "qsvm",
    "twolocal_1",
    "twolocal_2",
    # "twolocal_3",
    "vqe_1",
    "vqe_2",
    "vqe_3",
    "wstate",
]


def get_circuits(benchname: str, qubit_range: tuple[int, int]) -> list[QuantumCircuit]:
    abs_path = os.path.dirname(os.path.abspath(__file__))
    circuits_dir = os.path.join(abs_path, benchname)
    files = [
        fname
        for fname in os.listdir(circuits_dir)
        if fname.endswith(".qasm")
        and fname.split(".")[0].isdigit()
        and int(fname.split(".")[0]) in range(*qubit_range)
    ]
    if len(files) == 0:
        raise ValueError("No circuits found for the given qubit range.")
    circuits = sorted(
        [
            QuantumCircuit.from_qasm_file(os.path.join(circuits_dir, fname))
            for fname in files
        ],
        key=lambda x: x.num_qubits,
    )
    dags = [circuit_to_dag(circ) for circ in circuits]
    for dag in dags:
        dag.remove_all_ops_named("barrier")
    return [dag_to_circuit(dag) for dag in dags]
