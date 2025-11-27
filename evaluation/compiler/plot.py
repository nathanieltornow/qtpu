import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import benchkit as bk
import pandas as pd

from benchkit.plot.config import (
    double_column_width,
    single_column_width,
    colors,
    register_style,
    PlotStyle,
    get_style,
)

QTPU_LABEL = r"\textsc{qTPU}"
QAC_LABEL = r"\textsc{QAC}"


def plot_pareto_example(ax, qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot 1: Pareto frontier example for qnn 100q frac=0.5."""
    # Get QTPU data for qnn 100q frac=0.5
    qtpu_row = qtpu_df[
        (qtpu_df["config.bench"] == "qnn")
        & (qtpu_df["config.circuit_size"] == 100)
        & (qtpu_df["config.fraction"] == 0.5)
    ].iloc[0]

    # Get QAC data for same config
    qac_row = qac_df[
        (qac_df["config.bench"] == "qnn")
        & (qac_df["config.circuit_size"] == 100)
        & (qac_df["config.fraction"] == 0.5)
    ].iloc[0]

    # Extract Pareto frontier
    frontier = qtpu_row["result.pareto_frontier"]
    c_costs = [p["c_cost"] for p in frontier]
    errors = [p["max_error"] for p in frontier]

    # Sort by c_cost for line plot
    sorted_idx = np.argsort(c_costs)
    c_costs_sorted = [c_costs[i] for i in sorted_idx]
    errors_sorted = [errors[i] for i in sorted_idx]

    # Plot QTPU frontier
    ax.plot(
        c_costs_sorted,
        errors_sorted,
        "o-",
        color=colors()[0],
        markersize=6,
        linewidth=1.5,
        markeredgecolor="black",
        markeredgewidth=1,
        label=QTPU_LABEL,
        zorder=3,
    )

    # Fill area under frontier
    ax.fill_between(
        c_costs_sorted, errors_sorted, alpha=0.2, color=colors()[0], zorder=1
    )

    # Plot QAC single point
    qac_cost = qac_row["result.c_cost"]
    qac_error = qac_row["result.max_error"]
    ax.scatter(
        [qac_cost],
        [qac_error],
        marker="s",
        s=80,
        color=colors()[1],
        edgecolors="black",
        linewidths=1,
        label=QAC_LABEL,
        zorder=5,
    )

    ax.set_xlabel("Classical Cost [FLOPs]")
    ax.set_ylabel("Quantum Cost")
    ax.set_title(r"\textbf{(a) Pareto Frontier}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Yellow star at ideal point (0, 0)
    ax.scatter(
        [0],
        [0],
        marker="*",
        s=200,
        color="gold",
        edgecolors="black",
        linewidths=1,
        zorder=10,
        label="_nolegend_",
    )


def plot_error_comparison(ax, qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot 2: Lollipop chart showing each Pareto point for VQE-SU2."""
    sizes = [20, 40, 60, 80, 100, 120, 140]
    x_pos = np.arange(len(sizes))

    for i, size in enumerate(sizes):
        # Get QTPU data
        qtpu_row = qtpu_df[
            (qtpu_df["config.bench"] == "vqe_su2")
            & (qtpu_df["config.circuit_size"] == size)
            & (qtpu_df["config.fraction"] == 0.5)
        ]
        if qtpu_row.empty:
            continue
        qtpu_row = qtpu_row.iloc[0]

        # Get QAC data
        qac_row = qac_df[
            (qac_df["config.bench"] == "vqe_su2")
            & (qac_df["config.circuit_size"] == size)
            & (qac_df["config.fraction"] == 0.5)
        ]

        # QTPU Pareto points
        frontier = qtpu_row["result.pareto_frontier"]
        errors = sorted([p["max_error"] for p in frontier])
        min_err, max_err = min(errors), max(errors)

        # Draw vertical line (stem)
        ax.vlines(x_pos[i], min_err, max_err, colors=colors()[0], linewidth=2, zorder=2)

        # Draw dots for each Pareto point
        ax.scatter(
            [x_pos[i]] * len(errors),
            errors,
            color=colors()[0],
            s=50,
            edgecolors="white",
            linewidths=1,
            zorder=3,
            label=QTPU_LABEL if i == 0 else None,
        )

        # Draw QAC point if available
        if not qac_row.empty:
            qac_error = qac_row.iloc[0]["result.max_error"]
            ax.scatter(
                x_pos[i] + 0.15,
                qac_error,
                marker="s",
                s=70,
                color=colors()[1],
                edgecolors="black",
                linewidths=1,
                zorder=5,
                label=QAC_LABEL if i == 0 else None,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s}q" for s in sizes])
    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Quantum Cost")
    ax.set_title(r"\textbf{(b) Pareto Solutions}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")


def plot_compile_time_scalability(ax, qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot 3: Compile time scalability for VQE-SU2 at frac=0.5."""
    sizes = [20, 40, 60, 80, 100, 120, 140]

    qtpu_times = []
    qac_times = []

    for size in sizes:
        # Get QTPU compile time
        qtpu_row = qtpu_df[
            (qtpu_df["config.bench"] == "vqe_su2")
            & (qtpu_df["config.circuit_size"] == size)
            & (qtpu_df["config.fraction"] == 0.5)
        ]
        if not qtpu_row.empty:
            qtpu_times.append(qtpu_row.iloc[0]["result.compile_time"])
        else:
            qtpu_times.append(np.nan)

        # Get QAC compile time
        qac_row = qac_df[
            (qac_df["config.bench"] == "vqe_su2")
            & (qac_df["config.circuit_size"] == size)
            & (qac_df["config.fraction"] == 0.5)
        ]
        if not qac_row.empty:
            qac_times.append(qac_row.iloc[0]["result.compile_time"])
        else:
            qac_times.append(np.nan)

    # Plot lines
    ax.plot(
        sizes,
        qtpu_times,
        "o-",
        color=colors()[0],
        markersize=8,
        linewidth=2,
        markeredgecolor="black",
        markeredgewidth=1,
        label=QTPU_LABEL,
    )
    ax.plot(
        sizes,
        qac_times,
        "s-",
        color=colors()[1],
        markersize=8,
        linewidth=2,
        markeredgecolor="black",
        markeredgewidth=1,
        label=QAC_LABEL,
    )

    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Compile Time [s]")
    ax.set_title(r"\textbf{(c) Scalability} (lower is better $\downarrow$)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}q" for s in sizes])


@bk.pplot
def plot_compiler_quality(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot compiler quality comparison between QTPU and QAC.

    Args:
        qtpu_df: DataFrame with QTPU compilation results.
        qac_df: DataFrame with QAC compilation results.
    Returns:
        A BenchKit Plot object comparing the two compilers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.6))

    plot_pareto_example(axes[0], qtpu_df, qac_df)
    plot_error_comparison(axes[1], qtpu_df, qac_df)
    plot_compile_time_scalability(axes[2], qtpu_df, qac_df)

    return fig


if __name__ == "__main__":
    import matplotlib

    # Must disable TeX FIRST before any plot functions called
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    import matplotlib.pyplot as plt

    register_style("qtpu", PlotStyle(color=colors()[0], hatch="//"))
    register_style("qac", PlotStyle(color=colors()[1], hatch="\\\\"))

    # Load data
    qtpu_data = bk.load_log("logs/compiler/qtpu.jsonl")
    qac_data = bk.load_log("logs/compiler/qac.jsonl")

    qtpu_df = pd.DataFrame(qtpu_data)
    qac_df = pd.DataFrame(qac_data)

    fig = plot_compiler_quality(qtpu_df, qac_df)
