import argparse

from qiskit.circuit import QuantumCircuit

from qtpu.compiler.compiler import compile_circuit
from qtpu.compiler.partition_optimizer import NumQubitsOptimizer

from benchmarks import generate_benchmark, generate_benchmarks_range
from util import *


MAX_NUM_QUBITS = [20, 30, 40, 50, 60]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("min_qubits", type=int)
    parser.add_argument("max_qubits", type=int)
    parser.add_argument("benches", type=str, nargs="+")
    args = parser.parse_args()

    for bench in args.benches:
        circuits = generate_benchmarks_range(bench, args.min_qubits, args.max_qubits)

        for max_qubits in MAX_NUM_QUBITS:
            opt = NumQubitsOptimizer(max_qubits, "rm_1q")
            for circuit in circuits:

                htn = compile_circuit(circuit, opt)
                results = get_hybrid_tn_info(htn, None)
                results.update(
                    {
                        "name": bench,
                        "num_qubits": circuit.num_qubits,
                        "set_max_qubits": max_qubits,
                    }
                )
                append_to_csv(f"results/knit_cost.csv", results)


if __name__ == "__main__":
    main()
