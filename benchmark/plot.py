import sys
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import seaborn as sns

from benchmarks import pretty_names

FONTSIZE = 12
ISBETTER_FONTSIZE = FONTSIZE + 2
WIDE_FIGSIZE = (14, 2.8)
COLORS = sns.color_palette("pastel")

plt.rcParams.update({"font.size": FONTSIZE})

LINE_ARGS = {
    "markersize": 8,
    "markeredgewidth": 1.5,
    "markeredgecolor": "black",
    "linewidth": 2.5,
}


def _num_qubits_vs_cost(
    ax: plt.Axes, df: pd.DataFrame, benches: list[str], colors: dict[str, str]
) -> None:
    df = df.sort_values(by="num_qubits")

    # filter all rows where bruteforce_cost is < 0
    df = df[df["bruteforce_cost"] >= 0]
    df = df[df["set_max_qubits"] == 20]
    df = df[df["name"].isin(benches)]

    for name, group in df.groupby("name"):
        if name not in benches:
            continue
        ax.plot(
            group["num_qubits"],
            group["contract_cost"],
            "o-",
            label=name,
            color=colors[name],
            **LINE_ARGS,
        )
        ax.plot(
            group["num_qubits"],
            group["bruteforce_cost"],
            "^--",
            label=name + " bruteforce",
            color=colors[name],
            **LINE_ARGS,
        )


def _num_fragments_vs_cost(
    ax: plt.Axes, df: pd.DataFrame, benches: list[str], colors: dict[str, str]
) -> None:
    df = df[df["num_qubits"] == 100]
    # remove all rows where bruteforce_cost is < 0
    df = df[df["bruteforce_cost"] >= 0]
    df = df[df["name"].isin(benches)]

    # remove num_qubits column
    # df = df.drop(columns=["num_qubits"])
    for name, group in df.groupby("name"):
        
        group = group.sort_values(by="num_subcircuits")

        ax.plot(
            group["num_subcircuits"],
            group["contract_cost"],
            "o-",
            label=name,
            color=colors[name],
            **LINE_ARGS,
        )
        ax.plot(
            group["num_subcircuits"],
            group["bruteforce_cost"],
            "^--",
            label=name + " bruteforce",
            color=colors[name],
            **LINE_ARGS,
        )



def _circuit_size_vs_cost(
    ax: plt.Axes, df: pd.DataFrame, benches: list[str], colors: dict[str, str]
) -> None:
    df = df[df["num_qubits"] == 100]
    # remove all rows where bruteforce_cost is < 0
    df = df[df["bruteforce_cost"] >= 0]
    df = df[df["name"].isin(benches)]

    # remove num_qubits column
    # df = df.drop(columns=["num_qubits"])
    for name, group in df.groupby("name"):
        
        group = group.sort_values(by="set_max_qubits")

        ax.plot(
            group["set_max_qubits"],
            group["contract_cost"],
            "o-",
            label=name,
            color=colors[name],
            **LINE_ARGS,
        )
        ax.plot(
            group["set_max_qubits"],
            group["bruteforce_cost"],
            "^--",
            label=name + " bruteforce",
            color=colors[name],
            **LINE_ARGS,
        )



def plot_cost_bench(file_path: str) -> None:

    benches_1 = ["vqe_1", "hamsim_3", "twolocal_3"]
    benches_2 = ["qaoa_r2", "twolocal_2", "qsvm", "hamsim_3"]
    benches_3 = ["qaoa_r2", "twolocal_2", "qsvm", "hamsim_3"]
    
    color_dict = dict(zip(list(set(benches_1 + benches_2 + benches_3)), COLORS))

    df = pd.read_csv(file_path)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=WIDE_FIGSIZE, sharey=True)
    fig.subplots_adjust(wspace=0.08)
    
    ax0.set_title("(a)", fontweight="bold")
    ax0.set_xlabel("Number of qubits")
    ax0.set_ylabel("Contraction Cost [$10^x$ FLOPs]")
    

    ax1.set_title("(b)", fontweight="bold")
    ax1.set_xlabel("Number of subcircuits")

    ax2.set_title("(c)", fontweight="bold")
    ax2.set_xlabel("Max. subcircuit size")

    _num_qubits_vs_cost(ax0, df, set(benches_1), color_dict)
    
    _num_fragments_vs_cost(ax1, df, set(benches_2), color_dict)
    
    _circuit_size_vs_cost(ax2, df, set(benches_3), color_dict)

    our_line = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="-",
        label="qTPU",
        **LINE_ARGS,
    )
    base_line = mlines.Line2D(
        [],
        [],
        color="black",
        marker="^",
        linestyle="--",
        label="Bruteforce",
        **LINE_ARGS,
    )

    patches = []
    for label, color in color_dict.items():
        patches.append(mpatches.Patch(color=color, label=pretty_names[label]))

    fig.legend(
        handles=[our_line, base_line],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=10,
        frameon=False,
    )
    fig.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=10,
        frameon=False,
    )

    fig.savefig("knit_cost.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)
    args = parser.parse_args()

    match args.type:
        case "knit_cost":
            plot_cost_bench("results/knit_cost.csv")
