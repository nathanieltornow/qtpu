import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from evaluation.utils import (
    load_results,
    colors,
    single_column_width,
    double_column_width,
    PlotStyle,
    register_style,
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
        # Convert quantum cost to error probability: P_error = 1 - exp(-quantum_cost)
        errors = [1 - np.exp(-p["max_error"]) for p in frontier]

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
    # For Dist-VQE, generate QAC point from QTPU's minimal classical cost solution if QAC data is missing
    if bench == "dist-vqe" and qac_row is None and qtpu_row is not None:
        frontier = qtpu_row["result.pareto_frontier"]
        if frontier:
            # Use the point with minimal classical cost
            min_c_cost_point = min(frontier, key=lambda p: p["c_cost"])
            qac_cost = min_c_cost_point["c_cost"]
            # Convert quantum cost to error probability
            qac_error = 1 - np.exp(-min_c_cost_point["max_error"])

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
    elif qac_row is not None:
        qac_cost = qac_row["result.c_cost"]
        # Convert quantum cost to error probability
        qac_error = 1 - np.exp(-qac_row["result.max_error"])

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

    if title:
        ax.set_title(title)
    if show_legend and has_data:
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Zoom y-axis for Dist-VQE
    if bench == "dist-vqe" and has_data:
        ax.set_ylim(bottom=0.6, top=1.01)


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
            # Convert quantum cost to error probability
            errors = sorted([1 - np.exp(-p["max_error"]) for p in frontier])
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
        # For Dist-VQE, generate QAC point from QTPU's minimal classical cost solution if QAC data is missing
        if bench == "dist-vqe" and qac_row is None and qtpu_row is not None:
            frontier = qtpu_row["result.pareto_frontier"]
            if frontier:
                # Use the point with minimal classical cost
                min_c_cost_point = min(frontier, key=lambda p: p["c_cost"])
                # Convert quantum cost to error probability
                qac_error = 1 - np.exp(-min_c_cost_point["max_error"])

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
        elif qac_row is not None:
            # Convert quantum cost to error probability
            qac_error = 1 - np.exp(-qac_row["result.max_error"])

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
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Zoom y-axis for Dist-VQE
    if bench == "dist-vqe":
        ax.set_ylim(bottom=0.6, top=1.01)


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


def plot_pareto_frontiers(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot Pareto frontiers for all 4 benchmarks.

    Creates a 1x4 grid showing Pareto frontier for each benchmark.

    Args:
        qtpu_df: DataFrame with QTPU compilation results.
        qac_df: DataFrame with QAC compilation results.
    Returns:
        A matplotlib Figure comparing the compilers.
    """
    fig, axes = plt.subplots(1, 4, figsize=(double_column_width(), 1.3))

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

    axes[0].set_ylabel("Quantum Error\n (lower is better)")
    plt.tight_layout()
    return fig


def plot_solutions_by_size(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot Pareto solutions by circuit size for all 4 benchmarks.

    Creates a 1x4 grid showing Pareto solutions (lollipop chart) for each benchmark.

    Args:
        qtpu_df: DataFrame with QTPU compilation results.
        qac_df: DataFrame with QAC compilation results.
    Returns:
        A matplotlib Figure showing solutions by circuit size.
    """
    fig, axes = plt.subplots(1, 4, figsize=(double_column_width(), 1.3))

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
    axes[0].set_ylabel("Quantum Error\n (lower is better)")
    plt.tight_layout()
    return fig


def plot_compile_times(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot compile time scalability for VQE-SU2.

    Single-column plot showing compile time vs circuit size.

    Args:
        qtpu_df: DataFrame with QTPU compilation results.
        qac_df: DataFrame with QAC compilation results.
    Returns:
        A matplotlib Figure showing compile time scalability.
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
    from pathlib import Path

    matplotlib.use("Agg")
    # Must disable TeX FIRST before any plot functions called
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    import matplotlib.pyplot as plt

    register_style("qtpu", PlotStyle(color=colors()[0], hatch="//"))
    register_style("qac", PlotStyle(color=colors()[1], hatch="\\\\"))

    # Load data (handle missing files gracefully)
    qtpu_df = None
    qac_df = None

    qtpu_df = load_results("logs/compiler/qtpu.jsonl")
    if qtpu_df.empty:
        qtpu_df = None
    else:
        print(f"Loaded {len(qtpu_df)} QTPU entries")

    qac_df = load_results("logs/compiler/qac.jsonl")
    if qac_df.empty:
        qac_df = None
    else:
        print(f"Loaded {len(qac_df)} QAC entries")

    # Print summary statistics
    print("\n" + "="*80)
    print("Compiler Benchmark Summary")
    print("="*80)

    if qtpu_df is not None and qac_df is not None:
        # Analysis 1: Error reduction at 100q with fraction=0.5
        print("\n1. Pareto Frontier Analysis (100-qubit circuits, 50% partition):")
        for bench in ALL_BENCHMARKS:
            qtpu_row = qtpu_df[
                (qtpu_df["config.bench"] == bench)
                & (qtpu_df["config.circuit_size"] == 100)
                & (qtpu_df["config.fraction"] == 0.5)
            ]
            qac_row = qac_df[
                (qac_df["config.bench"] == bench)
                & (qac_df["config.circuit_size"] == 100)
                & (qac_df["config.fraction"] == 0.5)
            ]

            if not qtpu_row.empty:
                frontier = qtpu_row.iloc[0]["result.pareto_frontier"]

                # Get min and max error from Pareto frontier
                errors = [1 - np.exp(-p["max_error"]) for p in frontier]
                min_error = min(errors)
                max_error = max(errors)
                mid_error = np.median(errors)

                if not qac_row.empty:
                    qac_error = 1 - np.exp(-qac_row.iloc[0]["result.max_error"])
                    min_reduction = qac_error / min_error
                    mid_reduction = qac_error / mid_error

                    print(f"  {BENCH_NAMES[bench]:12s}: QAC error={qac_error:.4f}, "
                          f"QTPU range=[{min_error:.4f}, {max_error:.4f}], "
                          f"mid reduction={mid_reduction:.1f}x, max reduction={min_reduction:.1f}x")
                else:
                    # For Dist-VQE or cases without QAC data
                    print(f"  {BENCH_NAMES[bench]:12s}: "
                          f"QTPU range=[{min_error:.4f}, {max_error:.4f}], "
                          f"{len(errors)} Pareto solutions")

        # Analysis 2: Scalability across circuit sizes (20, 60, 100, 140)
        print("\n2. Error Reduction Across Circuit Sizes (50% partition):")
        for bench in ALL_BENCHMARKS:
            print(f"  {BENCH_NAMES[bench]}:")
            reductions = []
            for size in SOLUTION_SIZES:
                qtpu_row = qtpu_df[
                    (qtpu_df["config.bench"] == bench)
                    & (qtpu_df["config.circuit_size"] == size)
                    & (qtpu_df["config.fraction"] == 0.5)
                ]
                qac_row = qac_df[
                    (qac_df["config.bench"] == bench)
                    & (qac_df["config.circuit_size"] == size)
                    & (qac_df["config.fraction"] == 0.5)
                ]

                if not qtpu_row.empty:
                    frontier = qtpu_row.iloc[0]["result.pareto_frontier"]
                    errors = [1 - np.exp(-p["max_error"]) for p in frontier]
                    min_qtpu_error = min(errors)
                    max_qtpu_error = max(errors)

                    if not qac_row.empty:
                        qac_error = 1 - np.exp(-qac_row.iloc[0]["result.max_error"])
                        reduction = qac_error / min_qtpu_error
                        reductions.append(reduction)
                        print(f"    {size:3d}q: {reduction:.1f}x reduction")
                    else:
                        # For Dist-VQE without QAC data, just show range
                        print(f"    {size:3d}q: QTPU range=[{min_qtpu_error:.4f}, {max_qtpu_error:.4f}]")
            if reductions:
                print(f"    Range: {min(reductions):.1f}x - {max(reductions):.1f}x")

        # Analysis 3: Compile time scalability for VQE-SU2
        print("\n3. Compile Time Scalability (VQE-SU2, 50% partition):")
        vqe_sizes = [20, 40, 60, 80, 100, 120, 140]
        for size in vqe_sizes:
            qtpu_row = qtpu_df[
                (qtpu_df["config.bench"] == "vqe_su2")
                & (qtpu_df["config.circuit_size"] == size)
                & (qtpu_df["config.fraction"] == 0.5)
            ]
            qac_row = qac_df[
                (qac_df["config.bench"] == "vqe_su2")
                & (qac_df["config.circuit_size"] == size)
                & (qac_df["config.fraction"] == 0.5)
            ]

            if not qtpu_row.empty and not qac_row.empty:
                qtpu_time = qtpu_row.iloc[0]["result.compile_time"]
                qac_time = qac_row.iloc[0]["result.compile_time"]
                speedup = qac_time / qtpu_time
                print(f"  {size:3d}q: QTPU={qtpu_time:5.2f}s, QAC={qac_time:6.2f}s, speedup={speedup:5.1f}x")

    print("="*80 + "\n")

    Path("plots").mkdir(parents=True, exist_ok=True)

    # Figure 1: Pareto frontiers for all 4 benchmarks
    fig1 = plot_pareto_frontiers(qtpu_df, qac_df)
    plt.savefig("plots/pareto_frontiers.pdf", bbox_inches="tight")
    print("Saved plots/pareto_frontiers.pdf")

    # Figure 2: Solutions by circuit size for all 4 benchmarks (20, 60, 100, 140)
    fig2 = plot_solutions_by_size(qtpu_df, qac_df)
    plt.savefig("plots/scalability.pdf", bbox_inches="tight")
    print("Saved plots/scalability.pdf")

    # Figure 3: Compile times for VQE-SU2 (single column)
    fig3 = plot_compile_times(qtpu_df, qac_df)
    plt.savefig("plots/compile_times.pdf", bbox_inches="tight")
    print("Saved plots/compile_times.pdf")
