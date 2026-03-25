"""
Plot Error Mitigation Benchmark Results
========================================

Creates a 3-column plot showing:
1. Total time comparison (Naive vs Batch vs QTPU)
2. Time breakdown (Prep + Quantum + Classical)
3. Memory usage comparison
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils import (
    PlotStyle,
    colors,
    single_column_width,
    double_column_width,
    register_style,
    load_results,
)


QTPU_LABEL = r"\textsc{qTPU}"
BATCH_LABEL = r"\textsc{Batch}"
NAIVE_LABEL = r"\textsc{Naive}"


def load_and_prepare_data(
    naive_path: str, batch_path: str, qtpu_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load JSONL logs and convert to DataFrames."""
    naive_df = load_results(naive_path) if os.path.exists(naive_path) else pd.DataFrame()
    batch_df = load_results(batch_path) if os.path.exists(batch_path) else pd.DataFrame()
    qtpu_df = load_results(qtpu_path) if os.path.exists(qtpu_path) else pd.DataFrame()

    # Rename columns for convenience (JSONL logs use config.* and result.* prefixes)
    def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        rename_map = {
            "config.circuit_size": "circuit_size",
            "config.num_samples": "num_samples",
            "result.preparation_time": "preparation_time",
            "result.generation_time": "generation_time",
            "result.quantum_time": "quantum_time",
            "result.estimated_qpu_time": "estimated_qpu_time",
            "result.classical_time": "classical_time",
            "result.total_time": "total_time",
            "result.peak_memory": "peak_memory",
            "result.result_value": "result_value",
            "result.num_circuits": "num_circuits",
        }
        if "result.num_circuits_represented" in df.columns:
            rename_map["result.num_circuits_represented"] = "num_circuits_represented"
        return df.rename(columns=rename_map)

    naive_df = rename_cols(naive_df)
    batch_df = rename_cols(batch_df)
    qtpu_df = rename_cols(qtpu_df)

    # Convert memory from bytes to KB
    for df in [naive_df, batch_df, qtpu_df]:
        if not df.empty and "peak_memory" in df.columns:
            df["peak_memory_kb"] = df["peak_memory"] / 1024

    return naive_df, batch_df, qtpu_df


def plot_total_time(
    ax,
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    qtpu_df: pd.DataFrame,
    sample_size: int = 1000,
):
    """Plot total time comparison as grouped bar chart."""
    # Filter for specific sample size
    naive_scaled = naive_df[naive_df["num_samples"] == sample_size].sort_values("circuit_size") if not naive_df.empty else pd.DataFrame()
    batch_scaled = batch_df[batch_df["num_samples"] == sample_size].sort_values("circuit_size") if not batch_df.empty else pd.DataFrame()
    qtpu_scaled = qtpu_df[qtpu_df["num_samples"] == sample_size].sort_values("circuit_size") if not qtpu_df.empty else pd.DataFrame()

    # Use circuit sizes from whichever df has data
    circuit_sizes = []
    for df in [naive_scaled, batch_scaled, qtpu_scaled]:
        if not df.empty:
            circuit_sizes = df["circuit_size"].values
            break

    if len(circuit_sizes) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(circuit_sizes))
    width = 0.25

    # Naive bars
    if not naive_scaled.empty:
        ax.bar(
            x - width,
            naive_scaled["total_time"].values,
            width,
            label=NAIVE_LABEL,
            color=colors()[2],
            edgecolor="black",
            linewidth=1,
            hatch="xx",
        )

    # Batch bars
    if not batch_scaled.empty:
        ax.bar(
            x,
            batch_scaled["total_time"].values,
            width,
            label=BATCH_LABEL,
            color=colors()[3],
            edgecolor="black",
            linewidth=1,
            hatch="\\\\",
        )

    # QTPU bars
    if not qtpu_scaled.empty:
        ax.bar(
            x + width,
            qtpu_scaled["total_time"].values,
            width,
            label=QTPU_LABEL,
            color=colors()[0],
            edgecolor="black",
            linewidth=1,
            hatch="//",
        )

    ax.set_xlabel("Circuit Size [qubits]")
    ax.set_ylabel("Total Time [s]")
    ax.set_title(r"\textbf{(a) Total Time}" + "\n" + r"{(lower is better $\downarrow$)}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in circuit_sizes])
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")


def plot_time_breakdown(
    ax,
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    qtpu_df: pd.DataFrame,
    sample_size: int = 1000,
    circuit_size: int = 8,
):
    """Plot time breakdown as stacked bar chart for a specific config."""
    # Filter for specific sample size and circuit size
    def get_row(df, name):
        if df.empty:
            return None
        filtered = df[(df["num_samples"] == sample_size) & (df["circuit_size"] == circuit_size)]
        return filtered.iloc[0] if not filtered.empty else None

    naive_row = get_row(naive_df, "naive")
    batch_row = get_row(batch_df, "batch")
    qtpu_row = get_row(qtpu_df, "qtpu")

    approaches = []
    prep_times = []
    quantum_times = []
    classical_times = []

    if naive_row is not None:
        approaches.append(NAIVE_LABEL)
        prep_times.append(naive_row["preparation_time"])
        quantum_times.append(naive_row["quantum_time"])
        classical_times.append(naive_row.get("classical_time", 0))

    if batch_row is not None:
        approaches.append(BATCH_LABEL)
        prep_times.append(batch_row["preparation_time"])
        quantum_times.append(batch_row["quantum_time"])
        classical_times.append(batch_row.get("classical_time", 0))

    if qtpu_row is not None:
        approaches.append(QTPU_LABEL)
        prep_times.append(qtpu_row["preparation_time"])
        quantum_times.append(qtpu_row["quantum_time"])
        classical_times.append(qtpu_row.get("classical_time", 0))

    if not approaches:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(approaches))
    width = 0.6

    # Stacked bars
    ax.bar(x, prep_times, width, label="Preparation", color=colors()[0], edgecolor="black", linewidth=1)
    ax.bar(x, quantum_times, width, bottom=prep_times, label="Quantum", color=colors()[1], edgecolor="black", linewidth=1)
    bottoms = [p + q for p, q in zip(prep_times, quantum_times)]
    ax.bar(x, classical_times, width, bottom=bottoms, label="Classical", color=colors()[2], edgecolor="black", linewidth=1)

    ax.set_xlabel("Approach")
    ax.set_ylabel("Time [s]")
    ax.set_title(r"\textbf{(b) Time Breakdown}" + f"\n{{({circuit_size}q, {sample_size} samples)}}")
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend(loc="upper right", fontsize="small")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")


def plot_memory_usage(
    ax,
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    qtpu_df: pd.DataFrame,
    sample_size: int = 1000,
):
    """Plot memory usage comparison."""
    # Filter for specific sample size
    naive_scaled = naive_df[naive_df["num_samples"] == sample_size].sort_values("circuit_size") if not naive_df.empty else pd.DataFrame()
    batch_scaled = batch_df[batch_df["num_samples"] == sample_size].sort_values("circuit_size") if not batch_df.empty else pd.DataFrame()
    qtpu_scaled = qtpu_df[qtpu_df["num_samples"] == sample_size].sort_values("circuit_size") if not qtpu_df.empty else pd.DataFrame()

    # Use circuit sizes from whichever df has data
    circuit_sizes = []
    for df in [naive_scaled, batch_scaled, qtpu_scaled]:
        if not df.empty:
            circuit_sizes = df["circuit_size"].values
            break

    if len(circuit_sizes) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(circuit_sizes))
    width = 0.25

    # Naive bars
    if not naive_scaled.empty:
        ax.bar(
            x - width,
            naive_scaled["peak_memory_kb"].values,
            width,
            label=NAIVE_LABEL,
            color=colors()[2],
            edgecolor="black",
            linewidth=1,
            hatch="xx",
        )

    # Batch bars
    if not batch_scaled.empty:
        ax.bar(
            x,
            batch_scaled["peak_memory_kb"].values,
            width,
            label=BATCH_LABEL,
            color=colors()[3],
            edgecolor="black",
            linewidth=1,
            hatch="\\\\",
        )

    # QTPU bars
    if not qtpu_scaled.empty:
        ax.bar(
            x + width,
            qtpu_scaled["peak_memory_kb"].values,
            width,
            label=QTPU_LABEL,
            color=colors()[0],
            edgecolor="black",
            linewidth=1,
            hatch="//",
        )

    ax.set_xlabel("Circuit Size [qubits]")
    ax.set_ylabel("Peak Memory [KB]")
    ax.set_title(r"\textbf{(c) Memory Usage}" + "\n" + r"{(lower is better $\downarrow$)}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in circuit_sizes])
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")


def plot_error_mitigation_breakdown(
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    qtpu_df: pd.DataFrame,
    sample_size: int = 1000,
    circuit_size: int = 8,
):
    """Plot error mitigation breakdown comparison.

    Creates a 1x3 grid showing:
    - (a) Total time comparison
    - (b) Time breakdown (prep + quantum + classical)
    - (c) Memory usage

    Args:
        naive_df: DataFrame with naive benchmark results.
        batch_df: DataFrame with batch benchmark results.
        qtpu_df: DataFrame with QTPU benchmark results.
        sample_size: Number of samples for filtering.
        circuit_size: Circuit size for time breakdown plot.
    Returns:
        A matplotlib Figure object comparing the three approaches.
    """
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.6))

    plot_total_time(axes[0], naive_df, batch_df, qtpu_df, sample_size)
    plot_time_breakdown(axes[1], naive_df, batch_df, qtpu_df, sample_size, circuit_size)
    plot_memory_usage(axes[2], naive_df, batch_df, qtpu_df, sample_size)

    plt.tight_layout()
    return fig


def plot_sample_scaling(
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    qtpu_df: pd.DataFrame,
    circuit_size: int = 8,
    output_path: str = "plots/error_mitigation/sample_scaling.pdf",
):
    """Create a plot showing scaling with number of samples."""
    fig, ax = plt.subplots(figsize=(single_column_width(), 1.6))

    # Filter for a single circuit size
    naive_scaled = naive_df[naive_df["circuit_size"] == circuit_size].sort_values("num_samples") if not naive_df.empty else pd.DataFrame()
    batch_scaled = batch_df[batch_df["circuit_size"] == circuit_size].sort_values("num_samples") if not batch_df.empty else pd.DataFrame()
    qtpu_scaled = qtpu_df[qtpu_df["circuit_size"] == circuit_size].sort_values("num_samples") if not qtpu_df.empty else pd.DataFrame()

    # Line plots
    if not naive_scaled.empty:
        ax.plot(
            naive_scaled["num_samples"],
            naive_scaled["total_time"],
            marker="^",
            label=NAIVE_LABEL,
            color=colors()[2],
            linewidth=2,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )

    if not batch_scaled.empty:
        ax.plot(
            batch_scaled["num_samples"],
            batch_scaled["total_time"],
            marker="s",
            label=BATCH_LABEL,
            color=colors()[3],
            linewidth=2,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )

    if not qtpu_scaled.empty:
        ax.plot(
            qtpu_scaled["num_samples"],
            qtpu_scaled["total_time"],
            marker="o",
            label=QTPU_LABEL,
            color=colors()[0],
            linewidth=2,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Total Time [s]")
    ax.set_title(rf"\textbf{{Sample Scaling ({circuit_size}q)}}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    return fig


if __name__ == "__main__":
    # Disable TeX for local testing
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    register_style("qtpu", PlotStyle(color=colors()[0], hatch="//"))
    register_style("batch", PlotStyle(color=colors()[3], hatch="\\\\"))
    register_style("naive", PlotStyle(color=colors()[2], hatch="xx"))

    naive_path = "logs/error_mitigation/naive_breakdown.jsonl"
    batch_path = "logs/error_mitigation/batch_breakdown.jsonl"
    qtpu_path = "logs/error_mitigation/qtpu_breakdown.jsonl"

    naive_df, batch_df, qtpu_df = load_and_prepare_data(naive_path, batch_path, qtpu_path)

    print("Data loaded:")
    print(f"  Naive: {len(naive_df)} rows")
    print(f"  Batch: {len(batch_df)} rows")
    print(f"  QTPU: {len(qtpu_df)} rows")

    if not batch_df.empty:
        print("\nBatch results (1000 samples):")
        batch_1k = batch_df[batch_df["num_samples"] == 1000]
        print(batch_1k[["circuit_size", "total_time", "peak_memory_kb"]])

    if not qtpu_df.empty:
        print("\nQTPU results (1000 samples):")
        qtpu_1k = qtpu_df[qtpu_df["num_samples"] == 1000]
        print(qtpu_1k[["circuit_size", "total_time", "peak_memory_kb"]])

    # Main breakdown plot
    fig = plot_error_mitigation_breakdown(naive_df, batch_df, qtpu_df, sample_size=1000, circuit_size=8)
    plt.savefig("plots/error_mitigation/breakdown.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("plots/error_mitigation/breakdown.png", dpi=300, bbox_inches="tight")
    print("Saved plots/error_mitigation/breakdown.pdf")

    # Sample scaling plot
    plot_sample_scaling(naive_df, batch_df, qtpu_df, circuit_size=8)

    plt.show()
