import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib import gridspec

from data import SWAP_REDUCE_DATA, NOISE_SCALE_ALGIERS_DATA, DEP_MIN_DATA


FONTSIZE = 12


colors = sns.color_palette("pastel")
hatches = [
    "///",
    "\\\\",
    "o",
    "//",
    "--",
    "*",
    "//",
    "\\",
    "O",
    ".",
    "||",
    "/",
    "\\\\\\",
    "o",
    "///",
]


styles = {
    # "GHZ": (colors[0], hatches[0]),
    # "TL-1": (colors[1], hatches[1]),
    # "TL-2": (colors[2], hatches[2]),
    # "TL-3": (colors[3], hatches[3]),
    # "HS-1": (colors[4], hatches[4]),
    # "HS-2": (colors[5], hatches[5]),
    # "HS-3": (colors[6], hatches[6]),
    # "VQE-1": (colors[7], hatches[7]),
    # "VQE-2": (colors[8], hatches[8]),
    # "VQE-3": (colors[9], hatches[9]),
    # "QAOA-2": (colors[10], hatches[10]),
    # "QAOA-3": (colors[11], hatches[11]),
    # "QAOA-4": (colors[12], hatches[12]),
    # "QSVM": (colors[13], hatches[13]),
    # "W": (colors[14], hatches[14]),
}

plt.rcParams.update({"font.size": FONTSIZE})


def plot_bars(
    ax: plt.Axes,
    dataframes: list[pd.DataFrame],
    labels: list[str],
    xkey: str = "num_qubits",
    ykey: str = "relative",
    spacing: float = 1.0,
):
    assert len(dataframes) == len(labels) > 0

    ax.grid(axis="y", linestyle="--")

    all_xkeys = set()
    for df in dataframes:
        all_xkeys.update(set(df[xkey]))

    dataframes = [
        df.groupby(xkey)
        .agg({ykey: ["mean", "sem"]})
        .sort_values(by=["num_qubits"])
        .reset_index()
        for df in dataframes
    ]

    aggregated_dataframes = []
    for df in dataframes:
        # Set the xkey column as the index for the dataframe
        df = df.set_index(xkey)

        # Reindex the dataframe to include all values from ALL_XKEYS
        df = df.reindex(sorted(all_xkeys))

        # Fill missing values with 0 for the ykey column
        df[ykey] = df[ykey].fillna(0)

        # Reset the index to restore xkey as a separate column
        df = df.reset_index()

        # Store the aggregated dataframe in the dictionary
        aggregated_dataframes.append(df)

    dataframes = aggregated_dataframes

    nums_qubits = set(set(dataframes[0][xkey]))
    for df in dataframes[1:]:
        if nums_qubits != set(df[xkey]):
            raise ValueError("Dataframes must have the same xkey-s")

    nums_qubits = sorted(list(nums_qubits))  # type: ignore
    bar_width = spacing / (len(dataframes) + 1)
    x = np.arange(len(nums_qubits))

    for i, df in enumerate(dataframes):
        y = df[ykey]["mean"]
        yerr = df[ykey]["sem"]
        if np.isnan(yerr).any():
            yerr = None

        color, hatch = colors[i % len(colors)], hatches[i % len(hatches)]

        ax.bar(
            x + (i * bar_width),
            y,
            bar_width,
            hatch=hatch,
            label=labels[i],
            yerr=yerr,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            error_kw=dict(lw=2, capsize=3),
        )

    ax.set_xticks(x + ((len(dataframes) - 1) / 2) * bar_width)
    ax.set_xticklabels(nums_qubits)


def plot_relative(
    ax: plt.Axes,
    dataframes: list[pd.DataFrame],
    labels: list[str],
    xkey: str,
    xvalues: list[int],
    num_key: str,
    denom_key: str,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
):
    assert len(dataframes) == len(labels)

    dataframes = [df.loc[df[xkey].isin(xvalues)] for df in dataframes]
    dataframes = [df.loc[df[num_key] > 0.0] for df in dataframes]
    dataframes = [df.loc[df[denom_key] > 0.0] for df in dataframes]

    for df in dataframes:
        df["relative"] = df[num_key] / df[denom_key]

    plot_bars(
        ax,
        dataframes,
        labels,
        xkey=xkey,
        ykey="relative",
        spacing=0.95,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # plot line on 1
    ax.axhline(1, color="red", linestyle="-", linewidth=2)
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=FONTSIZE)


def plot_noisy_scale():
    dfs = [pd.read_csv(file) for file in NOISE_SCALE_ALGIERS_DATA.values()]
    labels = list(NOISE_SCALE_ALGIERS_DATA.keys())

    fig = plt.figure(figsize=(6.5, 2.6))
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    plot_relative(
        ax0,
        dfs,
        labels,
        xkey="num_qubits",
        xvalues=[6, 10, 14],
        num_key="depth",
        denom_key="depth_base",
        xlabel="Number of Qubits",
        ylabel="Relative Depth",
        title="(a) Depth (lower is better)",
    )
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(base=0.5))

    plot_relative(
        ax=ax1,
        dataframes=dfs,
        labels=labels,
        xkey="num_qubits",
        xvalues=[6, 10, 14],
        num_key="h_fid",
        denom_key="h_fid_base",
        xlabel="Number of Qubits",
        ylabel="Relative Fidelity",
        title="(b) Fidelity (higher is better)",
    )

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(base=0.2))

    ax1.set_ylim(bottom=0.9)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=5)

    plt.tight_layout()
    plt.savefig("figures/noisy_scale/plot.pdf", bbox_inches="tight")


def plot_swap_reduce():
    dfs = [pd.read_csv(file) for file in SWAP_REDUCE_DATA.values()]
    labels = list(SWAP_REDUCE_DATA.keys())

    fig = plt.figure(figsize=(6.5, 2.6))
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    plot_relative(
        ax0,
        dfs,
        labels,
        xkey="num_qubits",
        xvalues=[6, 16, 24],
        num_key="depth",
        denom_key="depth_base",
        xlabel="Number of Qubits",
        ylabel="Relative Depth",
        title="(a) Depth (lower is better)",
    )
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(base=0.5))

    plot_relative(
        ax=ax1,
        dataframes=dfs,
        labels=labels,
        xkey="num_qubits",
        xvalues=[6, 16, 24],
        num_key="num_cnots",
        denom_key="num_cnots_base",
        xlabel="Number of Qubits",
        ylabel="Relative CNOTs",
        title="(b) CNOTs (lower is better)",
    )
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(base=0.5))

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=5)

    plt.tight_layout()
    plt.savefig("figures/swap_reduce/plot.pdf", bbox_inches="tight")


def plot_dep_min():
    from data import DEP_MIN_DATA_2

    dfs = [pd.read_csv(file) for file in DEP_MIN_DATA_2.values()]
    labels = list(DEP_MIN_DATA_2.keys())

    fig = plt.figure(figsize=(12.5, 2.8))
    
    xvalues=[8, 16, 24]

    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    # ax2 = fig.add_subplot(gs[0, 2])

    # plot_relative(
    #     ax=ax0,
    #     dataframes=dfs,
    #     labels=labels,
    #     xkey="num_qubits",
    #     xvalues=xvalues,
    #     num_key="num_cnots",
    #     denom_key="num_cnots_base",
    #     xlabel="Number of Qubits",
    #     ylabel="Relative CNOTs",
    #     title="(a) CNOTs (lower is better)",
    # )
    
    plot_relative(
        ax=ax1,
        dataframes=dfs,
        labels=labels,
        xkey="num_qubits",
        xvalues=xvalues,
        num_key="depth",
        denom_key="depth_base",
        xlabel="Number of Qubits",
        ylabel="Relative Depth",
        title="(b) Depth (lower is better)",
    )
    
    plot_relative(
        ax=ax0,
        dataframes=dfs,
        labels=labels,
        xkey="num_qubits",
        xvalues=xvalues,
        num_key="num_deps",
        denom_key="num_deps_base",
        xlabel="Number of Qubits",
        ylabel="Relative Qubit Deps",
        title="(a) Qubit Dependencies (lower is better)",
    )

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(base=0.5))


    handles, labels = ax0.get_legend_handles_labels()
    nrows = 1
    ncols = len(labels) // nrows
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1 * nrows), ncol=ncols)

    plt.tight_layout()
    plt.savefig("figures/dep_min/overall_depmin.pdf", bbox_inches="tight")


# plot_noisy_scale()
plot_dep_min()