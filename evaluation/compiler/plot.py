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

# Benchmark display names
BENCH_NAMES = {
    "qnn": "QNN",
    "wstate": "W-State",
    "vqe_su2": "VQE-SU2",
    "dist-vqe": "Dist-VQE",
}

# All 4 benchmarks
ALL_BENCHMARKS = ["qnn", "wstate", "vqe_su2", "dist-vqe"]
CIRCUIT_SIZES = [20, 40, 60, 80, 100, 120, 140]
SOLUTION_SIZES = [20, 60, 100, 140]


def plot_pareto_frontier(
    ax,
    qtpu_df: pd.DataFrame,
    qac_df: pd.DataFrame,
    bench: str,
    circuit_size: int = 100,
    fraction: float = 0.5,
    title: str = None,
    show_legend: bool = True,
):
    """Plot Pareto frontier for a specific benchmark."""
    has_data = False
    
    # Get QTPU data
    qtpu_row = None
    if qtpu_df is not None and not qtpu_df.empty:
        qtpu_filtered = qtpu_df[
            (qtpu_df["config.bench"] == bench)
            & (qtpu_df["config.circuit_size"] == circuit_size)
            & (qtpu_df["config.fraction"] == fraction)
        ]
        if not qtpu_filtered.empty:
            qtpu_row = qtpu_filtered.iloc[0]

    # Get QAC data for same config
    qac_row = None
    if qac_df is not None and not qac_df.empty:
        qac_filtered = qac_df[
            (qac_df["config.bench"] == bench)
            & (qac_df["config.circuit_size"] == circuit_size)
            & (qac_df["config.fraction"] == fraction)
        ]
        if not qac_filtered.empty:
            qac_row = qac_filtered.iloc[0]

    # Plot QTPU frontier
    if qtpu_row is not None:
        frontier = qtpu_row["result.pareto_frontier"]
        c_costs = [p["c_cost"] for p in frontier]
        errors = [p["max_error"] for p in frontier]

        # Sort by c_cost for line plot
        sorted_idx = np.argsort(c_costs)
        c_costs_sorted = [c_costs[i] for i in sorted_idx]
        errors_sorted = [errors[i] for i in sorted_idx]

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
        has_data = True

    # Plot QAC single point
    if qac_row is not None:
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
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Classical Cost [FLOPs]")
    ax.set_ylabel("Quantum Cost")
    if title:
        ax.set_title(title)
    if show_legend and has_data:
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_pareto_solutions(
    ax,
    qtpu_df: pd.DataFrame,
    qac_df: pd.DataFrame,
    bench: str,
    sizes: list = None,
    fraction: float = 0.5,
    title: str = None,
    show_legend: bool = True,
):
    """Lollipop chart showing Pareto points for a specific benchmark."""
    if sizes is None:
        sizes = SOLUTION_SIZES
    x_pos = np.arange(len(sizes))

    qtpu_plotted = False
    qac_plotted = False

    for i, size in enumerate(sizes):
        # Get QTPU data
        qtpu_row = None
        if qtpu_df is not None and not qtpu_df.empty:
            qtpu_filtered = qtpu_df[
                (qtpu_df["config.bench"] == bench)
                & (qtpu_df["config.circuit_size"] == size)
                & (qtpu_df["config.fraction"] == fraction)
            ]
            if not qtpu_filtered.empty:
                qtpu_row = qtpu_filtered.iloc[0]

        # Get QAC data
        qac_row = None
        if qac_df is not None and not qac_df.empty:
            qac_filtered = qac_df[
                (qac_df["config.bench"] == bench)
                & (qac_df["config.circuit_size"] == size)
                & (qac_df["config.fraction"] == fraction)
            ]
            if not qac_filtered.empty:
                qac_row = qac_filtered.iloc[0]

        # QTPU Pareto points
        if qtpu_row is not None:
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
                label=QTPU_LABEL if not qtpu_plotted else None,
            )
            qtpu_plotted = True

        # Draw QAC point if available
        if qac_row is not None:
            qac_error = qac_row["result.max_error"]
            ax.scatter(
                x_pos[i] + 0.15,
                qac_error,
                marker="s",
                s=70,
                color=colors()[1],
                edgecolors="black",
                linewidths=1,
                zorder=5,
                label=QAC_LABEL if not qac_plotted else None,
            )
            qac_plotted = True

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s}q" for s in sizes])
    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Quantum Cost")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")


def plot_compile_time_scalability(
    ax,
    qtpu_df: pd.DataFrame,
    qac_df: pd.DataFrame,
    bench: str,
    sizes: list = None,
    fraction: float = 0.5,
    title: str = None,
    show_legend: bool = True,
):
    """Compile time scalability for a specific benchmark."""
    if sizes is None:
        sizes = CIRCUIT_SIZES

    qtpu_times = []
    qac_times = []

    for size in sizes:
        # Get QTPU compile time
        if qtpu_df is not None and not qtpu_df.empty:
            qtpu_row = qtpu_df[
                (qtpu_df["config.bench"] == bench)
                & (qtpu_df["config.circuit_size"] == size)
                & (qtpu_df["config.fraction"] == fraction)
            ]
            if not qtpu_row.empty:
                qtpu_times.append(qtpu_row.iloc[0]["result.compile_time"])
            else:
                qtpu_times.append(np.nan)
        else:
            qtpu_times.append(np.nan)

        # Get QAC compile time
        if qac_df is not None and not qac_df.empty:
            qac_row = qac_df[
                (qac_df["config.bench"] == bench)
                & (qac_df["config.circuit_size"] == size)
                & (qac_df["config.fraction"] == fraction)
            ]
            if not qac_row.empty:
                qac_times.append(qac_row.iloc[0]["result.compile_time"])
            else:
                qac_times.append(np.nan)
        else:
            qac_times.append(np.nan)

    # Plot lines (only if there's valid data)
    if not all(np.isnan(qtpu_times)):
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
    if not all(np.isnan(qac_times)):
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
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}q" for s in sizes])


@bk.pplot
def plot_pareto_frontiers(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot Pareto frontiers for all 4 benchmarks.

    Creates a 1x4 grid showing Pareto frontier for each benchmark.

    Args:
        qtpu_df: DataFrame with QTPU compilation results.
        qac_df: DataFrame with QAC compilation results.
    Returns:
        A BenchKit Plot object comparing the compilers.
    """
    fig, axes = plt.subplots(1, 4, figsize=(double_column_width(), 1.6))

    for i, bench in enumerate(ALL_BENCHMARKS):
        plot_pareto_frontier(
            axes[i],
            qtpu_df,
            qac_df,
            bench,
            circuit_size=100,
            fraction=0.5,
            title=rf"\textbf{{({chr(ord('a') + i)}) {BENCH_NAMES[bench]}}}",
            show_legend=(i == 0),
        )

    plt.tight_layout()
    return fig


@bk.pplot
def plot_solutions_by_size(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot Pareto solutions by circuit size for all 4 benchmarks.

    Creates a 1x4 grid showing Pareto solutions (lollipop chart) for each benchmark.

    Args:
        qtpu_df: DataFrame with QTPU compilation results.
        qac_df: DataFrame with QAC compilation results.
    Returns:
        A BenchKit Plot object showing solutions by circuit size.
    """
    fig, axes = plt.subplots(1, 4, figsize=(double_column_width(), 1.6))

    for i, bench in enumerate(ALL_BENCHMARKS):
        plot_pareto_solutions(
            axes[i],
            qtpu_df,
            qac_df,
            bench,
            sizes=SOLUTION_SIZES,
            fraction=0.5,
            title=rf"\textbf{{({chr(ord('a') + i)}) {BENCH_NAMES[bench]}}}",
            show_legend=(i == 0),
        )

    plt.tight_layout()
    return fig


@bk.pplot
def plot_compile_times(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot compile time scalability for VQE-SU2.

    Single-column plot showing compile time vs circuit size.

    Args:
        qtpu_df: DataFrame with QTPU compilation results.
        qac_df: DataFrame with QAC compilation results.
    Returns:
        A BenchKit Plot object showing compile time scalability.
    """
    fig, ax = plt.subplots(1, 1, figsize=(single_column_width(), 1.6))

    plot_compile_time_scalability(
        ax,
        qtpu_df,
        qac_df,
        "vqe_su2",
        sizes=CIRCUIT_SIZES,
        fraction=0.5,
        title=rf"\textbf{{Compile Time Scalability}}",
        show_legend=True,
    )

    plt.tight_layout()
    return fig





if __name__ == "__main__":
    import matplotlib
    import os

    # Must disable TeX FIRST before any plot functions called
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    import matplotlib.pyplot as plt

    register_style("qtpu", PlotStyle(color=colors()[0], hatch="//"))
    register_style("qac", PlotStyle(color=colors()[1], hatch="\\\\"))

    # Load data (handle missing files gracefully)
    qtpu_df = None
    qac_df = None

    if os.path.exists("logs/compiler/qtpu.jsonl"):
        qtpu_data = bk.load_log("logs/compiler/qtpu.jsonl")
        qtpu_df = pd.DataFrame(qtpu_data)
        print(f"Loaded {len(qtpu_df)} QTPU entries")

    if os.path.exists("logs/compiler/qac.jsonl"):
        qac_data = bk.load_log("logs/compiler/qac.jsonl")
        qac_df = pd.DataFrame(qac_data)
        print(f"Loaded {len(qac_df)} QAC entries")

    # Figure 1: Pareto frontiers for all 4 benchmarks
    fig1 = plot_pareto_frontiers(qtpu_df, qac_df)

    # Figure 2: Solutions by circuit size for all 4 benchmarks (20, 60, 100, 140)
    fig2 = plot_solutions_by_size(qtpu_df, qac_df)

    # Figure 3: Compile times for VQE-SU2 (single column)
    fig3 = plot_compile_times(qtpu_df, qac_df)

    plt.show()
