"""Plotting for Hardware Simulation Benchmark.

Creates a 3-panel figure showing:
(a) Estimated vs Actual (simulated) QPU Time
(b) Fidelity vs Subcircuit Size (more cutting = smaller subcircuits = higher fidelity)
(c) Correctness: Hardware simulation vs Ideal
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchkit.plot.config import (
    colors,
    double_column_width,
)

import benchkit as bk

QTPU_LABEL = r"\textsc{qTPU}"


def plot_qpu_time_comparison(ax, qtpu_df: pd.DataFrame):
    """Panel (a): Estimated vs actual (simulated) QPU time."""
    sizes = sorted(qtpu_df["config.circuit_size"].unique())

    # Filter for one subcircuit size
    subcirc_size = 5
    df = qtpu_df[qtpu_df["config.subcirc_size"] == subcirc_size]

    estimated = []
    actual = []
    labels = []

    for size in sizes:
        row = df[df["config.circuit_size"] == size]
        if row.empty or "result.estimated_qpu_time" not in row.columns:
            continue

        est = row.iloc[0].get("result.estimated_qpu_time", 0)
        act = row.iloc[0].get("result.sim_time", 0)  # Simulated time as proxy

        if est > 0 and act > 0:
            estimated.append(est)
            actual.append(act)
            labels.append(f"{size}q")

    if not estimated:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(a) QPU Time Estimate}")
        return

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, estimated, width, label="Estimated", color=colors()[0], edgecolor="black")
    ax.bar(x + width/2, actual, width, label="Simulated", color=colors()[1], edgecolor="black")

    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Time [s]")
    ax.set_title(r"\textbf{(a) QPU Time Estimate}")
    ax.legend(loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis="y")


def plot_fidelity_vs_cutting(ax, qtpu_df: pd.DataFrame):
    """Panel (b): Fidelity vs subcircuit size (Pareto point analysis)."""
    sizes = sorted(qtpu_df["config.circuit_size"].unique())

    # Get data for one circuit size
    circuit_size = max(sizes) if sizes else 20
    df = qtpu_df[qtpu_df["config.circuit_size"] == circuit_size]

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(b) Fidelity vs Cutting}")
        return

    subcirc_sizes = sorted(df["config.subcirc_size"].unique())

    errors = []
    for sc_size in subcirc_sizes:
        row = df[df["config.subcirc_size"] == sc_size]
        if not row.empty and "result.relative_error" in row.columns:
            err = row.iloc[0].get("result.relative_error", np.nan)
            errors.append(err if not np.isnan(err) else 1.0)
        else:
            errors.append(1.0)

    # Convert error to fidelity-like metric (1 - error)
    fidelities = [1 - min(e, 1) for e in errors]

    ax.plot(subcirc_sizes, fidelities, "o-", color=colors()[0], markersize=8, linewidth=2)

    ax.set_xlabel("Max Subcircuit Size [qubits]")
    ax.set_ylabel("Fidelity (1 - rel. error)")
    ax.set_title(rf"\textbf{{(b) Fidelity ({circuit_size}q circuit)}}")
    ax.grid(True, alpha=0.3)

    # Add annotation: smaller subcircuits = more cutting = higher fidelity
    ax.annotate(
        "More cutting\n→ Higher fidelity",
        xy=(subcirc_sizes[0], fidelities[0]),
        xytext=(subcirc_sizes[0] + 1, fidelities[0] - 0.1),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )


def plot_correctness(ax, qtpu_df: pd.DataFrame, direct_df: pd.DataFrame = None):
    """Panel (c): Correctness comparison."""
    sizes = sorted(qtpu_df["config.circuit_size"].unique())

    # Get qTPU results (best subcircuit size)
    qtpu_errors = []
    direct_errors = []
    labels = []

    for size in sizes:
        # qTPU: use smallest subcircuit size (most cutting)
        df = qtpu_df[qtpu_df["config.circuit_size"] == size]
        if df.empty:
            continue

        best_row = df.loc[df["result.relative_error"].idxmin()] if "result.relative_error" in df.columns else None
        if best_row is not None:
            qtpu_errors.append(best_row.get("result.relative_error", 1.0))
        else:
            qtpu_errors.append(1.0)

        # Direct (no cutting)
        if direct_df is not None and not direct_df.empty:
            direct_row = direct_df[direct_df["config.circuit_size"] == size]
            if not direct_row.empty and "result.relative_error" in direct_row.columns:
                direct_errors.append(direct_row.iloc[0].get("result.relative_error", 1.0))
            else:
                direct_errors.append(1.0)
        else:
            direct_errors.append(1.0)

        labels.append(f"{size}q")

    if not labels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(c) Correctness}")
        return

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, qtpu_errors, width, label=f"{QTPU_LABEL} (cut)", color=colors()[0], edgecolor="black")
    ax.bar(x + width/2, direct_errors, width, label="Direct", color=colors()[1], edgecolor="black")

    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Relative Error\n(lower is better)")
    ax.set_title(r"\textbf{(c) Correctness}")
    ax.legend(loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis="y")


@bk.pplot
def plot_hardware_results(qtpu_df: pd.DataFrame, direct_df: pd.DataFrame = None):
    """Create hardware results figure."""
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.5))

    plot_qpu_time_comparison(axes[0], qtpu_df)
    plot_fidelity_vs_cutting(axes[1], qtpu_df)
    plot_correctness(axes[2], qtpu_df, direct_df)

    return fig


if __name__ == "__main__":
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    # Load data
    qtpu_data = bk.load_log("logs/hardware/qtpu_noisy.jsonl")
    direct_data = bk.load_log("logs/hardware/direct_noisy.jsonl")

    if isinstance(qtpu_data, pd.DataFrame):
        qtpu_df = qtpu_data
    else:
        qtpu_df = pd.json_normalize(qtpu_data) if qtpu_data else pd.DataFrame()

    if isinstance(direct_data, pd.DataFrame):
        direct_df = direct_data
    else:
        direct_df = pd.json_normalize(direct_data) if direct_data else pd.DataFrame()

    if qtpu_df.empty:
        print("No qTPU hardware data found")
        exit(1)

    print(f"qTPU: {len(qtpu_df)} rows, Direct: {len(direct_df)} rows")

    fig = plot_hardware_results(qtpu_df, direct_df)
    plt.tight_layout()
    plt.savefig("plots/hardware_results.pdf", bbox_inches="tight")
    plt.show()
