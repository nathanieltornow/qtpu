import argparse

from qiskit.circuit import QuantumCircuit

from qvm.compiler.compiler import compile_circuit
from qvm.compiler.optimizer import NumQubitsOptimizer

from benchmarks import generate_benchmark, generate_benchmarks_range
from util import *

MAX_NUM_QUBITS = [20, 30, 40, 50, 60]


def knit_cost_benchmark(circuit: QuantumCircuit, max_qubits: int) -> dict:
    optimizer = NumQubitsOptimizer(max_qubits, "rm_1q")
    hybrid_tn = compile_circuit(circuit, optimizer)
    return get_hybrid_tn_info(hybrid_tn, None)


def fidelity_benchmark(circuits: list[QuantumCircuit]) -> None:
    pass


def _run_bench(circuit: QuantumCircuit, bench_type: str) -> dict:
    match bench_type:
        case "knit_cost":
            for max_qubits in MAX_NUM_QUBITS:
                results = knit_cost_benchmark(circuit)
        case _:
            raise ValueError(f"Unknown benchmark type: {args.type}")
    return results


if __name__ == "__main__":
    # parse benchmark-type, a list of benchmark-names, min_qubits, max_qubits
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)
    parser.add_argument("min_qubits", type=int)
    parser.add_argument("max_qubits", type=int)
    parser.add_argument("benches", type=str, nargs="+")
    args = parser.parse_args()

    # for bench in args.benches:
    #     circuits = generate_benchmarks_range(bench, args.min_qubits, args.max_qubits)
    #     for circuit in circuits:
    #         results = _run_bench(circuit, args.type)
    #         results.update({"name": bench, "num_qubits": circuit.num_qubits})
    #         append_to_csv(f"results/knit_cost.csv", results)
