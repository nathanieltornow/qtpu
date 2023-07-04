import matplotlib.pyplot as plt
import string
import math
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd

from util import calculate_figure_size, plot_lines


sns.set_theme(style="whitegrid", color_codes=True)

plt.rcParams.update({"font.size": 12})


def plot_noisy_scale_abs(
    dataframes: list[pd.DataFrame],
    titles: list[str],
    output_file: str = "noisy_scale.pdf",
) -> None:
    # plotting the absolute fidelities
    fig = plt.figure(figsize=calculate_figure_size(1, len(dataframes)))
    gs = gridspec.GridSpec(nrows=1, ncols=len(dataframes))

    axis = [fig.add_subplot(gs[0, i]) for i in range(len(dataframes))]

    axis[0].set_ylabel("Hellinger Fidelity")
    for ax in axis:
        ax.set_xlabel("# of Qubits")

    for let, title, ax, df in zip(string.ascii_lowercase, titles, axis, dataframes):
        plot_lines(ax, ["depth", "depth_base"], ["QVM", "Baseline"], [df])
        ax.legend()
        ax.set_title(f"({let}) {title}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")


def main():
    noisy_scale_files = {
        "GHZ": "bench/results/noisy_scale/ibm_perth_vs_ibmq_guadalupe/ghz.csv",
        "Two Local 1": "bench/results/noisy_scale/ibm_perth_vs_ibmq_guadalupe/hamsim_1.csv",
    }
    plot_noisy_scale_abs(
        [pd.read_csv(file) for file in noisy_scale_files.values()],
        list(noisy_scale_files.keys()),
    )
    
    
if __name__ == "__main__":
    main()
