import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import benchkit as bk
from benchkit.plot.config import (
    get_style,
    register_style,
    PlotStyle,
    single_column_width,
    colors,
)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def prepare_df(df: pd.DataFrame, prefix: str):
    """Flatten dict columns and compute derived fields."""
    # Compute q_cost = max(qtensor_errors)
    df[f"result.{prefix}.q_cost"] = df[f"result.{prefix}.qtensor_errors"].apply(max)

    # For convenience:
    df[f"size"] = df["config.circuit_size"]
    df[f"bench"] = df["config.bench"]

    # Max width of subcircuits
    df[f"result.{prefix}.max_width"] = df[f"result.{prefix}.qtensor_widths"].apply(max)

    # Compile time
    df[f"result.{prefix}.compile_time"] = df[f"result.{prefix}.compile_time"]

    # Classical cost
    df[f"result.{prefix}.c_cost"] = df[f"result.{prefix}.c_cost"]

    return df


# ------------------------------------------------------------
# Plot 1: Pareto frontier
# ------------------------------------------------------------
def plot_pareto(ax, qtpu_df, qac_df):
    ax.scatter(
        qtpu_df["result.qtpu.c_cost"],
        qtpu_df["result.qtpu.q_cost"],
        c=get_style("qtpu").color,
        marker="o",
        s=20,
        label="QTPU",
    )
    ax.scatter(
        qac_df["result.qac.c_cost"],
        qac_df["result.qac.q_cost"],
        c=get_style("qac").color,
        marker="s",
        s=40,
        label="QAC",
    )

    ax.set_xlabel("Classical Contraction Cost (log)")
    ax.set_ylabel("Quantum Noise Cost (max error)")
    ax.set_xscale("log")
    ax.set_ylim(0, 0.7)
    ax.legend()
    ax.set_title("Pareto Frontier: q_cost vs c_cost")


# ------------------------------------------------------------
# Plot 2: Max width vs size
# ------------------------------------------------------------
def plot_width_vs_size(ax, qtpu_df, qac_df):
    # Mean across hyperparameters
    qtp = qtpu_df.groupby("size")["result.qtpu.max_width"].mean()
    qac = qac_df.groupby("size")["result.qac.max_width"].mean()

    ax.plot(
        qtp.index, qtp.values, label="QTPU", marker="o", color=get_style("qtpu").color
    )
    ax.plot(
        qac.index, qac.values, label="QAC", marker="s", color=get_style("qac").color
    )

    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Max Subcircuit Width")
    ax.set_title("Subcircuit Width vs Circuit Size")
    ax.legend()


# ------------------------------------------------------------
# Plot 3: Maximum error vs size
# ------------------------------------------------------------
def plot_error_vs_size(ax, qtpu_df, qac_df):
    qtp = qtpu_df.groupby("size")["result.qtpu.q_cost"].mean()
    qac = qac_df.groupby("size")["result.qac.q_cost"].mean()

    ax.plot(
        qtp.index, qtp.values, label="QTPU", marker="o", color=get_style("qtpu").color
    )
    ax.plot(
        qac.index, qac.values, label="QAC", marker="s", color=get_style("qac").color
    )

    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Quantum Noise Cost (max error)")
    ax.set_ylim(0, 0.7)
    ax.set_title("Noise vs Circuit Size")
    ax.legend()


@bk.pplot
def plot_compile_time(qtpu_df, qac_df):

    # merge on circuit_size
    # qtpu_df = qtpu_df[qtpu_df["config.num_trials"] == 10]
    merged_df = pd.merge(
        qtpu_df,
        qac_df,
        on=["config.circuit_size", "config.bench"],
        suffixes=(".qtpu", ".qac"),
    )
    merged_df.rename(
        columns={"result.qtpu.compile_time": "qtpu", "result.qac.compile_time": "qac"},
        inplace=True,
    )
    merged_df = merged_df[merged_df["config.max_qubits"] == 30]
    merged_df = merged_df[merged_df["config.num_trials"] == 100]
    merged_df = merged_df[merged_df["config.bench"] == "vqe_su2"]

    merged_df = merged_df[
        merged_df["config.circuit_size"].isin(list(range(10, 110, 30)))
    ]

    fig, ax = plt.subplots(figsize=(single_column_width(), 1.7))
    bk.plot.bar_comparison(
        ax=ax,
        results=merged_df,
        keys=["qtpu", "qac"],
        group_key="config.circuit_size",
        error="std",
    )
    return fig


def compute_hqs(df, prefix, alpha=0.8):
    df = df.copy()
    df["q"] = df[f"result.{prefix}.qtensor_errors"].apply(max)
    df["c"] = df[f"result.{prefix}.c_cost"]

    c_max = df["c"].max()

    df["c_norm"] = np.log(df["c"] + 1) / np.log(c_max + 1)

    df["hqs"] = 1.0 / (alpha * df["q"] + (1 - alpha) * df["c_norm"])

    return df


def compute_efficiency(df, prefix):
    df = df.copy()
    df["q"] = df[f"result.{prefix}.qtensor_errors"].apply(max)
    df["c"] = df[f"result.{prefix}.c_cost"]
    df["efficiency"] = (1 - df["q"]) / np.log(df["c"] + 2)
    return df


def pareto_frontier(df, x_col, y_col):
    pts = df[[x_col, y_col]].to_numpy()
    n = len(pts)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if keep[i]:
            # a point dominates if both metrics <=
            dominates = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
            keep[dominates] = False

    return df[keep]


@bk.pplot
def plot_pareto_frontiers(qtpu_df, qac_df, bench="vqe_su2"):

    qtpu = qtpu_df[qtpu_df["config.bench"] == bench].copy()
    qac = qac_df[qac_df["config.bench"] == bench].copy()

    qtpu["q_cost"] = qtpu["result.qtpu.qtensor_errors"].apply(max)
    qtpu["c_cost"] = qtpu["result.qtpu.c_cost"]

    qac["q_cost"] = qac["result.qac.qtensor_errors"].apply(max)
    qac["c_cost"] = qac["result.qac.c_cost"]

    sizes = sorted(qtpu["config.circuit_size"].unique())
    fig, axs = plt.subplots(
        len(sizes),
        1,
        figsize=(single_column_width(), 2 * len(sizes)),
        sharex=True,
        sharey=True,
    )

    if len(sizes) == 1:
        axs = [axs]

    for ax, size in zip(axs, sizes):
        qtpu_s = qtpu[qtpu["config.circuit_size"] == size]

        # Compute true Pareto frontier
        frontier = pareto_frontier(qtpu_s, "c_cost", "q_cost")

        ax.scatter(
            frontier["c_cost"],
            frontier["q_cost"],
            color=get_style("qtpu").color,
            marker="o",
            s=20,
            label="QTPU",
        )

        qac_s = qac[qac["config.circuit_size"] == size]
        if len(qac_s):
            ax.scatter(
                qac_s["c_cost"],
                qac_s["q_cost"],
                color=get_style("qac").color,
                marker="s",
                s=40,
                label="QAC",
            )

        ax.set_title(f"Circuit size = {size}")
        ax.set_xscale("log")
        ax.set_ylabel("q_error")

    axs[-1].set_xlabel("Classical cost (log)")
    axs[0].legend()

    fig.tight_layout()
    return fig


@bk.pplot
def plot_qcost(qtpu_df, qac_df):
    fig, ax = plt.subplots(figsize=(single_column_width(), 1.7))
    qtpu_df = compute_efficiency(qtpu_df, "qtpu")
    qac_df = compute_efficiency(qac_df, "qac")
    # merge on circuit_size
    merged_df = pd.merge(
        qtpu_df,
        qac_df,
        on=["config.circuit_size", "config.bench"],
        suffixes=(".qtpu", ".qac"),
    )
    merged_df.rename(
        columns={"efficiency.qtpu": "qtpu", "efficiency.qac": "qac"},
        inplace=True,
    )
    merged_df = merged_df[merged_df["config.max_qubits"] == 10]
    merged_df = merged_df[merged_df["config.num_trials"] == 100]
    merged_df = merged_df[merged_df["config.bench"] == "qnn"]

    merged_df = merged_df[
        merged_df["config.circuit_size"].isin(list(range(10, 110, 30)))
    ]

    bk.plot.bar_comparison(
        ax=ax,
        results=merged_df,
        keys=["qtpu", "qac"],
        group_key="config.circuit_size",
        error="std",
    )
    return fig


# ------------------------------------------------------------
# Combined 2×2 figure
# ------------------------------------------------------------
@bk.pplot
def plot_compiler_results(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):

    # Prepare DFs
    qtpu_df = prepare_df(qtpu_df, "qtpu")
    qac_df = prepare_df(qac_df, "qac")

    # Filter extremely huge c_cost if needed
    qtpu_df = qtpu_df[qtpu_df["result.qtpu.c_cost"] < 1e5]
    qac_df = qac_df[qac_df["result.qac.c_cost"] < 1e5]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    plot_pareto(axs[0, 0], qtpu_df, qac_df)
    plot_width_vs_size(axs[0, 1], qtpu_df, qac_df)
    plot_error_vs_size(axs[1, 0], qtpu_df, qac_df)
    plot_compile_time(axs[1, 1], qtpu_df)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# Run manually
# ------------------------------------------------------------
if __name__ == "__main__":
    register_style("qtpu", PlotStyle(marker="o", color=colors()[0]))
    register_style("qac", PlotStyle(marker="s", color=colors()[1]))

    qtpu_df = bk.load_log("logs/compile/qtpu.jsonl")
    qac_df = bk.load_log("logs/compile/qac.jsonl")

    # plot_compiler_results(qtpu_df, qac_df)
    plot_compile_time(qtpu_df, qac_df)
    plot_qcost(qtpu_df, qac_df)
    plot_pareto_frontiers(qtpu_df, qac_df, bench="vqe_su2")