import matplotlib.pyplot as plt
import string
import math
import os
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd

from util import calculate_figure_size, plot_lines
from data import SWAP_REDUCE_DATA, DEP_MIN_DATA, NOISE_SCALE_ALGIERS_DATA


sns.set_theme(style="whitegrid", color_codes=True)
sns.color_palette("deep")

plt.rcParams.update({"font.size": 12})


def plot_swap_reduce() -> None:
    dfs = [pd.read_csv(file) for file in SWAP_REDUCE_DATA.values()]
    titles = list(SWAP_REDUCE_DATA.keys())

    plot_dataframes(
        dataframes=dfs,
        keys=["num_cnots", "num_cnots_base"],
        labels=["SWAP Reduced", "Baseline"],
        titles=titles,
        ylabel="Number of CNOTs",
        xlabel="Number of Qubits",
        output_file="figures/swap_reduce/cnot.pdf",
    )
    plot_dataframes(
        dataframes=dfs,
        keys=["depth", "depth_base"],
        labels=["SWAP Reduced", "Baseline"],
        titles=titles,
        ylabel="Circuit Depth",
        xlabel="Number of Qubits",
        output_file="figures/swap_reduce/depth.pdf",
    )


def plot_dep_min() -> None:
    dfs = [pd.read_csv(file) for file in DEP_MIN_DATA.values()]
    titles = list(DEP_MIN_DATA.keys())

    plot_dataframes(
        dataframes=dfs,
        keys=["num_cnots", "num_cnots_base"],
        labels=["Dep Min", "Baseline"],
        titles=titles,
        ylabel="Number of CNOTs",
        xlabel="Number of Qubits",
        output_file="figures/dep_min/cnot.pdf",
    )
    plot_dataframes(
        dataframes=dfs,
        keys=["depth", "depth_base"],
        labels=["Dep Min", "Baseline"],
        titles=titles,
        ylabel="Circuit Depth",
        xlabel="Number of Qubits",
        output_file="figures/dep_min/depth.pdf",
    )


def plot_noisy_scale() -> None:
    dfs = [pd.read_csv(file) for file in NOISE_SCALE_ALGIERS_DATA.values()]
    titles = list(NOISE_SCALE_ALGIERS_DATA.keys())

    plot_dataframes(
        dataframes=dfs,
        keys=["num_cnots", "num_cnots_base"],
        labels=["Dep Min", "Baseline"],
        titles=titles,
        ylabel="Number of CNOTs",
        xlabel="Number of Qubits",
        output_file="figures/noisy_scale/algiers_cnot.pdf",
    )
    plot_dataframes(
        dataframes=dfs,
        keys=["depth", "depth_base"],
        labels=["Dep Min", "Baseline"],
        titles=titles,
        ylabel="Circuit Depth",
        xlabel="Number of Qubits",
        output_file="figures/noisy_scale/algiers_depth.pdf",
    )
    plot_dataframes(
        dataframes=dfs,
        keys=["tv_fid", "tv_fid_base"],
        labels=["Ours", "Baseline"],
        titles=titles,
        ylabel="Fidelity",
        xlabel="Number of Qubits",
        output_file="figures/noisy_scale/algiers_fid.pdf",
        nrows=2,
    )


def plot_dataframes(
    dataframes: list[pd.DataFrame],
    keys: list[str],
    labels: list[str],
    titles: list[str],
    ylabel: str,
    xlabel: str,
    output_file: str = "noisy_scale.pdf",
    nrows: int = 1,
) -> None:
    ncols = len(dataframes) // nrows + len(dataframes) % nrows
    # plotting the absolute fidelities
    fig = plt.figure(figsize=calculate_figure_size(nrows, ncols))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

    axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

    for i, ax in enumerate(axis):
        if i % nrows == 0:
            ax.set_ylabel(ylabel=ylabel)
        ax.set_xlabel(xlabel=xlabel)

    for let, title, ax, df in zip(string.ascii_lowercase, titles, axis, dataframes):
        plot_lines(ax, keys, labels, [df])
        ax.legend()
        ax.set_title(f"({let}) {title}", fontsize=12, fontweight="bold")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")


def plot_relative(
    dataframes: list[pd.DataFrame],
    num_key: str,
    denom_key: str,
    label: str,
    titles: list[str],
    ylabel: str,
    xlabel: str,
    output_file: str = "noisy_scale.pdf",
):
    pass


def main():
    print("Plotting figures...")
    print("Plotting swap reduce...")
    plot_swap_reduce()
    print("Plotting dep min...")
    plot_dep_min()
    print("Plotting noisy scale...")
    plot_noisy_scale()


if __name__ == "__main__":
    main()
