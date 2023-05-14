import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLORS = ['#E59866', '#B5838D', '#6D9197', '#B5BFA1', '#D6B4E0', '#E9A3C9', '#C5CBE3', '#A5D2BA', '#F7CAC9', '#9CAFB7']

def plot_cut_comparison(ax, csv_name: str, labels: list[str], caption: str):
    df = pd.read_csv(csv_name)
    group_df = df.groupby(["num_qubits"]).agg([np.mean, np.std]).reset_index()
    group_df = group_df.sort_values(by=["num_qubits"])
    x = np.arange(len(group_df))  # the label locations
    width = 1 / (len(group_df) + 1)  # the width of the bars
    if len(labels) == 2:
        multiplier = 0.5
    elif len(labels) == 3:
        multiplier = -0.0
    else:
        raise ValueError("Invalid number of labels.")

    for i, label in enumerate(labels):
        offset = width * multiplier
        ax.bar(
            x + offset,
            group_df[label]["mean"],
            yerr=group_df[label]["std"],
            color=COLORS[i],
            width=width,
            label=label,
            ecolor="black",
            capsize=1,
            edgecolor="black",
        )

        for i, mean in enumerate(group_df[label]["mean"]):
            if mean == 0:
                ax.text(
                    x[i] + offset,
                    0.022,
                    "x",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="red",
                    fontweight="bold",
                )
        #     # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_yscale("log")

    ax.set_xticks(x + width, group_df["num_qubits"])
    ax.set_title(caption)


fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0])

import sys

plot_cut_comparison(
    ax,
    sys.argv[1],
    [
        # "wire_cut_time",
        # "gate_cut_time",
        # "optimal_cut_time",
        # "gate_num_fragments",
        # "wire_num_fragments",
        # "optimal_num_fragments",
        # "gate_cuts",
        # "wire_cuts",
        "gate_overhead",
        "wire_overhead",
        "optimal_overhead",
    ],
    "QAOA",
)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower left", ncol=3, )
plt.show()
