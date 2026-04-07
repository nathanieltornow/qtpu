"""Plotting for Combined Workload Benchmark.

Creates a 3-panel figure showing:
(a) Circuit Count: qTPU vs Baseline (log scale)
(b) E2E Time Breakdown: compile + qpu + classical (stacked bars)
(c) Speedup Summary: speedup factors across configurations
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchkit.plot.config import (
    PlotStyle,
    colors,
    double_column_width,
    register_style,
)

import benchkit as bk

QTPU_LABEL = r"\textsc{qTPU}"
BASELINE_LABEL = r"Baseline"


def plot_circuit_count(ax, qtpu_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """Panel (a): Circuit count comparison (log scale)."""
    sizes = sorted(qtpu_df["config.circuit_size"].unique())

    x = np.arange(len(sizes))
    width = 0.35

    qtpu_counts = []
    baseline_counts = []

    for size in sizes:
        qtpu_row = qtpu_df[qtpu_df["config.circuit_size"] == size]
        baseline_row = baseline_df[baseline_df["config.circuit_size"] == size]

        if not qtpu_row.empty:
            qtpu_counts.append(qtpu_row.iloc[0]["result.num_circuits"])
        else:
            qtpu_counts.append(1)

        if not baseline_row.empty and "result.num_circuits" in baseline_row.columns:
            baseline_counts.append(baseline_row.iloc[0]["result.num_circuits"])
        else:
            baseline_counts.append(1)

    ax.bar(
        x - width / 2,
        qtpu_counts,
        width,
        label=QTPU_LABEL,
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )

    ax.bar(
        x + width / 2,
        baseline_counts,
        width,
        label=BASELINE_LABEL,
        color=colors()[1],
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )

    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Circuits Generated\n(log scale)")
    ax.set_title(r"\textbf{(a) Circuit Count}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in sizes])

    # Add ratio annotations
    for i, (q, b) in enumerate(zip(qtpu_counts, baseline_counts)):
        if b > 0 and q > 0:
            ratio = b / q
            ax.annotate(
                f"{ratio:.0f}×",
                xy=(i, max(q, b) * 1.5),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )


def plot_time_breakdown(ax, qtpu_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """Panel (b): E2E time breakdown (stacked bars)."""
    sizes = sorted(qtpu_df["config.circuit_size"].unique())

    x = np.arange(len(sizes) * 2)  # qTPU and baseline for each size
    width = 0.6

    labels = []
    compile_times = []
    qpu_times = []
    classical_times = []

    for size in sizes:
        # qTPU
        qtpu_row = qtpu_df[qtpu_df["config.circuit_size"] == size]
        if not qtpu_row.empty:
            compile_times.append(qtpu_row.iloc[0].get("result.compile_time", 0))
            qpu_times.append(qtpu_row.iloc[0].get("result.quantum_time", 0))
            classical_times.append(qtpu_row.iloc[0].get("result.classical_time", 0))
        else:
            compile_times.append(0)
            qpu_times.append(0)
            classical_times.append(0)
        labels.append(f"{QTPU_LABEL}\n{size}q")

        # Baseline
        baseline_row = baseline_df[baseline_df["config.circuit_size"] == size]
        if not baseline_row.empty:
            compile_times.append(baseline_row.iloc[0].get("result.compile_time", 0))
            qpu_times.append(baseline_row.iloc[0].get("result.quantum_time", 0))
            classical_times.append(baseline_row.iloc[0].get("result.classical_time", 0))
        else:
            compile_times.append(0)
            qpu_times.append(0)
            classical_times.append(0)
        labels.append(f"{BASELINE_LABEL}\n{size}q")

    # Stacked bars
    ax.bar(x, compile_times, width, label="Compile", color=colors()[0], edgecolor="black")
    ax.bar(
        x,
        qpu_times,
        width,
        bottom=compile_times,
        label="QPU (est.)",
        color=colors()[1],
        edgecolor="black",
    )
    ax.bar(
        x,
        classical_times,
        width,
        bottom=[c + q for c, q in zip(compile_times, qpu_times)],
        label="Classical",
        color=colors()[2],
        edgecolor="black",
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Time [s]")
    ax.set_title(r"\textbf{(b) E2E Time Breakdown}")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)


def plot_speedup_summary(ax, qtpu_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """Panel (c): Speedup summary."""
    sizes = sorted(qtpu_df["config.circuit_size"].unique())

    circuit_speedups = []
    time_speedups = []
    code_speedups = []

    for size in sizes:
        qtpu_row = qtpu_df[qtpu_df["config.circuit_size"] == size]
        baseline_row = baseline_df[baseline_df["config.circuit_size"] == size]

        if qtpu_row.empty or baseline_row.empty:
            continue

        # Circuit count speedup
        q_circuits = qtpu_row.iloc[0].get("result.num_circuits", 1)
        b_circuits = baseline_row.iloc[0].get("result.num_circuits", 1)
        circuit_speedups.append(b_circuits / max(q_circuits, 1))

        # Time speedup
        q_time = (
            qtpu_row.iloc[0].get("result.compile_time", 0)
            + qtpu_row.iloc[0].get("result.quantum_time", 0)
            + qtpu_row.iloc[0].get("result.classical_time", 0)
        )
        b_time = (
            baseline_row.iloc[0].get("result.compile_time", 0)
            + baseline_row.iloc[0].get("result.quantum_time", 0)
            + baseline_row.iloc[0].get("result.classical_time", 0)
        )
        time_speedups.append(b_time / max(q_time, 0.001))

        # Code size speedup
        q_code = qtpu_row.iloc[0].get("result.total_code_lines", 1)
        b_code = baseline_row.iloc[0].get("result.total_code_lines", 1)
        code_speedups.append(b_code / max(q_code, 1))

    if not circuit_speedups:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(c) Speedup Summary}")
        return

    x = np.arange(3)
    values = [np.mean(circuit_speedups), np.mean(time_speedups), np.mean(code_speedups)]
    labels = ["Circuits", "E2E Time", "Code Size"]

    bars = ax.bar(x, values, color=[colors()[0], colors()[1], colors()[2]], edgecolor="black")

    ax.set_ylabel("Speedup (×)")
    ax.set_title(r"\textbf{(c) Speedup Summary}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value annotations
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.0f}×",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontweight="bold",
        )


@bk.pplot
def plot_combined_workload(qtpu_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """Create combined workload comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.5))

    plot_circuit_count(axes[0], qtpu_df, baseline_df)
    plot_time_breakdown(axes[1], qtpu_df, baseline_df)
    plot_speedup_summary(axes[2], qtpu_df, baseline_df)

    return fig


if __name__ == "__main__":
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    # Load data
    qtpu_data = bk.load_log("logs/combined/qtpu.jsonl")
    baseline_data = bk.load_log("logs/combined/baseline.jsonl")

    if isinstance(qtpu_data, pd.DataFrame):
        qtpu_df = qtpu_data
    else:
        qtpu_df = pd.json_normalize(qtpu_data) if qtpu_data else pd.DataFrame()

    if isinstance(baseline_data, pd.DataFrame):
        baseline_df = baseline_data
    else:
        baseline_df = pd.json_normalize(baseline_data) if baseline_data else pd.DataFrame()

    if qtpu_df.empty:
        print("No qTPU data found")
        exit(1)

    print(f"qTPU: {len(qtpu_df)} rows, Baseline: {len(baseline_df)} rows")

    fig = plot_combined_workload(qtpu_df, baseline_df)
    plt.tight_layout()
    plt.savefig("plots/combined_workload.pdf", bbox_inches="tight")
    plt.show()
