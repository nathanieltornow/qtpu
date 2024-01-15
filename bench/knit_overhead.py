import os
import pandas as pd
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit

from qvm.cutter.girvan_newman import GirvanNewmanCutter
from qvm.virtual_circuit import VirtualCircuit

from circuits import get_circuits
from util import get_base_knit_overhead, get_knit_overhead


RESULT_FILE = "knit_overhead.csv"


def run(benchname: str, num_qubits: tuple[int, int]) -> None:
    circuits = get_circuits(benchname, num_qubits)

    df = pd.DataFrame(
        columns=[
            "benchname",
            "num_qubits",
            "knit_overhead",
            "base_knit_overhead",
        ],
    )
    for circuit in circuits:
        cut_circuit = GirvanNewmanCutter(100).run(circuit)
        virtual_circuit = VirtualCircuit(cut_circuit)

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "benchname": [benchname],
                        "num_qubits": [circuit.num_qubits],
                        "knit_overhead": [get_knit_overhead(virtual_circuit)],
                        "base_knit_overhead": [get_base_knit_overhead(virtual_circuit)],
                    }
                ),
            ],
            ignore_index=True,
        )

    df.to_csv(
        RESULT_FILE, mode="a", header=not os.path.exists(RESULT_FILE), index=False
    )

    df = df.groupby(["benchname", "num_qubits"]).mean()


def main():
    run("twolocal_1", (18, 31))


if __name__ == "__main__":
    main()
