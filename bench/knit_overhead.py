import os
import pandas as pd
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit

import qvm
from qvm.compiler import MetisCutter

from circuits import get_circuits
from bench.util import get_virtual_circuit_info, append_to_csv
from plot import postprocess_barplot


RESULT_FILE = "results/knit_overhead.csv"


def run(bechnames: list[str], num_qubits: int, num_fragments: int) -> None:
    for benchname in bechnames:
        circuit = get_circuits(benchname, (num_qubits, num_qubits + 1))[0]
        cut_circuit = MetisCutter(num_fragments).run(circuit)
        virtual_circuit = qvm.VirtualCircuit(cut_circuit)

        info = get_virtual_circuit_info(virtual_circuit).to_dict()
        info["benchname"] = benchname
        info["num_qubits"] = num_qubits
        append_to_csv(RESULT_FILE, info)


name_table = {
    "twolocal_1": "TL-1",
    "twolocal_2": "TL-2",
    "qsvm": "QSVM",
    "wstate": "WState",
    "hamsim_1": "HS-1",
    "hamsim_2": "HS-2",
    "vqe_1": "VQE-1",
    "vqe_2": "VQE-2",
}


def make_figure() -> None:
    df = pd.read_csv(RESULT_FILE)

    # df["rel"] = df["naive_knit_cost"] / df["knit_cost"]

    df = df[["benchname", "n_fragments", "naive_knit_cost"]]

    df["benchname"] = df["benchname"].apply(lambda x: name_table[x])

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 2.8))
    df = df.groupby(["benchname", "n_fragments"]).mean().unstack()

    import seaborn as sns

    df.plot.bar(
        ax=ax,
        rot=0,
        color=sns.color_palette("pastel", 5),
        edgecolor="black",
        legend=False,
        xlabel="",
        width=0.8,
    )

    # set x to log scale
    ax.set_yscale("log")
    # ax.set_xticklabels([2, 3, 4, 5, 6])
    # ax.set_title("Knit Overhead vs. CKT", fontweight="bold")
    ax.set_ylabel("Knitting Cost (FLOPs)")

    postprocess_barplot(ax)

    handles, labels = ax.get_legend_handles_labels()

    # legend at the bottom of entire fig
    fig.legend(
        handles,
        ["2 fragments", "3 fragments", "4 fragments", "5 fragments"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.15),
    )

    fig.tight_layout()
    fig.savefig("total_cost.pdf", bbox_inches="tight")
    plt.show()


def main():
    benches = [
        "twolocal_1",
        "twolocal_2",
        "qsvm",
        "wstate",
        "hamsim_1",
        "hamsim_2",
        "vqe_1",
        "vqe_2",
    ]

    for nf in [2, 3, 4, 5]:
        run(benches, 100, nf)


if __name__ == "__main__":
    # plt.rcParams.update({"font.size": 14})

    make_figure()
    # main()
