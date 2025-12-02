"""
Plot Hybrid ML Inference Benchmark Results
==========================================

Creates publication-quality plots showing:
1. Total time comparison by circuit size
2. Time breakdown (Prep + Quantum + Classical) 
3. Scaling with batch size
4. Scaling with feature dimension
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchkit as bk
from benchkit.plot.config import (
    PlotStyle,
    colors,
    single_column_width,
    double_column_width,
    register_style,
)


QTPU_LABEL = r"\textsc{qTPU}"
BATCH_LABEL = r"\textsc{Batch}"
NAIVE_LABEL = r"\textsc{Naive}"

# Consistent hatches across all plots
HATCHES = {"naive": "xxx", "batch": "\\\\\\", "heinsum": "///"}


def load_and_prepare_data(
    naive_path: str, batch_path: str, heinsum_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load JSONL logs and convert to DataFrames."""
    naive_df = bk.load_log(naive_path) if os.path.exists(naive_path) else pd.DataFrame()
    batch_df = bk.load_log(batch_path) if os.path.exists(batch_path) else pd.DataFrame()
    heinsum_df = bk.load_log(heinsum_path) if os.path.exists(heinsum_path) else pd.DataFrame()

    # Rename columns for convenience (benchkit uses config.* and result.* prefixes)
    def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        rename_map = {
            "config.circuit_size": "circuit_size",
            "config.feature_dim": "feature_dim",
            "config.batch_size": "batch_size",
            "result.preparation_time": "preparation_time",
            "result.quantum_time": "quantum_time",
            "result.classical_time": "classical_time",
            "result.total_time": "total_time",
            "result.peak_memory": "peak_memory",
            "result.num_circuits": "num_circuits",
        }
        return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    naive_df = rename_cols(naive_df)
    batch_df = rename_cols(batch_df)
    heinsum_df = rename_cols(heinsum_df)

    # Convert memory from bytes to KB
    for df in [naive_df, batch_df, heinsum_df]:
        if not df.empty and "peak_memory" in df.columns:
            df["peak_memory_kb"] = df["peak_memory"] / 1024

    return naive_df, batch_df, heinsum_df


def plot_total_time_by_circuit_size(
    ax,
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    batch_size: int = 100,
    feature_dim: int = 4,
):
    """Plot total time comparison as grouped bar chart by circuit size."""
    
    def filter_df(df, name):
        if df.empty:
            return pd.DataFrame()
        filtered = df[
            (df["batch_size"] == batch_size) & 
            (df["feature_dim"] == feature_dim)
        ].sort_values("circuit_size")
        return filtered
    
    naive_filt = filter_df(naive_df, "naive")
    batch_filt = filter_df(batch_df, "batch")
    heinsum_filt = filter_df(heinsum_df, "heinsum")

    # Get circuit sizes from whichever df has data
    circuit_sizes = []
    for df in [naive_filt, batch_filt, heinsum_filt]:
        if not df.empty:
            circuit_sizes = sorted(df["circuit_size"].unique())
            break

    if len(circuit_sizes) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(circuit_sizes))
    width = 0.25

    # Naive bars
    if not naive_filt.empty:
        times = [naive_filt[naive_filt["circuit_size"] == s]["total_time"].values[0] 
                 if s in naive_filt["circuit_size"].values else 0 for s in circuit_sizes]
        ax.bar(x - width, times, width, label=NAIVE_LABEL,
               color=colors()[2], edgecolor="black", linewidth=0.5, hatch=HATCHES["naive"])

    # Batch bars
    if not batch_filt.empty:
        times = [batch_filt[batch_filt["circuit_size"] == s]["total_time"].values[0]
                 if s in batch_filt["circuit_size"].values else 0 for s in circuit_sizes]
        ax.bar(x, times, width, label=BATCH_LABEL,
               color=colors()[3], edgecolor="black", linewidth=0.5, hatch=HATCHES["batch"])

    # HEinsum bars
    if not heinsum_filt.empty:
        times = [heinsum_filt[heinsum_filt["circuit_size"] == s]["total_time"].values[0]
                 if s in heinsum_filt["circuit_size"].values else 0 for s in circuit_sizes]
        ax.bar(x + width, times, width, label=QTPU_LABEL,
               color=colors()[0], edgecolor="black", linewidth=0.5, hatch=HATCHES["heinsum"])

    ax.set_xlabel("Circuit Size [qubits]")
    ax.set_ylabel("Total Time [s]")
    ax.set_title(r"\textbf{(a) Total Time}" + "\n" + r"\textit{lower is better $\downarrow$}", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in circuit_sizes])
    ax.legend(loc="upper left", fontsize=6)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")


def plot_time_breakdown(
    ax,
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    circuit_size: int = 8,
    batch_size: int = 100,
    feature_dim: int = 4,
):
    """Plot time breakdown as stacked bar chart."""
    
    def get_row(df):
        if df.empty:
            return None
        filtered = df[
            (df["circuit_size"] == circuit_size) &
            (df["batch_size"] == batch_size) &
            (df["feature_dim"] == feature_dim)
        ]
        return filtered.iloc[0] if not filtered.empty else None

    naive_row = get_row(naive_df)
    batch_row = get_row(batch_df)
    heinsum_row = get_row(heinsum_df)

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

    if heinsum_row is not None:
        approaches.append(QTPU_LABEL)
        prep_times.append(heinsum_row["preparation_time"])
        quantum_times.append(heinsum_row["quantum_time"])
        classical_times.append(heinsum_row.get("classical_time", 0))

    if not approaches:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(approaches))
    width = 0.6

    # Stacked bars (quantum at bottom, prep, classical on top)
    ax.bar(x, quantum_times, width, label="Quantum (QPU)",
           color=colors()[1], edgecolor="black", linewidth=0.5)
    ax.bar(x, prep_times, width, bottom=quantum_times, label="Prep (CPU)",
           color=colors()[0], edgecolor="black", linewidth=0.5, hatch="///")
    bottoms = [q + p for q, p in zip(quantum_times, prep_times)]
    ax.bar(x, classical_times, width, bottom=bottoms, label="Classical",
           color=colors()[2], edgecolor="black", linewidth=0.5, hatch="...")

    ax.set_xlabel("Approach")
    ax.set_ylabel("Time [s]")
    ax.set_title(r"\textbf{(b) Time Breakdown}" + f"\n{{({circuit_size}q, batch={batch_size})}}", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=7)
    ax.legend(loc="upper right", fontsize=5)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")


def plot_batch_scaling(
    ax,
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    circuit_size: int = 8,
    feature_dim: int = 4,
):
    """Plot total time scaling with batch size."""
    
    def filter_and_sort(df):
        if df.empty:
            return pd.DataFrame()
        return df[
            (df["circuit_size"] == circuit_size) &
            (df["feature_dim"] == feature_dim)
        ].sort_values("batch_size")
    
    naive_filt = filter_and_sort(naive_df)
    batch_filt = filter_and_sort(batch_df)
    heinsum_filt = filter_and_sort(heinsum_df)

    if not naive_filt.empty:
        ax.plot(naive_filt["batch_size"], naive_filt["total_time"],
                "^-", label=NAIVE_LABEL, color=colors()[2], linewidth=2,
                markersize=8, markeredgecolor="black", markeredgewidth=1)

    if not batch_filt.empty:
        ax.plot(batch_filt["batch_size"], batch_filt["total_time"],
                "s-", label=BATCH_LABEL, color=colors()[3], linewidth=2,
                markersize=8, markeredgecolor="black", markeredgewidth=1)

    if not heinsum_filt.empty:
        ax.plot(heinsum_filt["batch_size"], heinsum_filt["total_time"],
                "o-", label=QTPU_LABEL, color=colors()[0], linewidth=2,
                markersize=8, markeredgecolor="black", markeredgewidth=1)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Total Time [s]")
    ax.set_title(r"\textbf{(c) Batch Scaling}" + f"\n{{({circuit_size}q)}}", fontsize=8)
    ax.legend(loc="upper left", fontsize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def plot_feature_scaling(
    ax,
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    circuit_size: int = 8,
    batch_size: int = 100,
):
    """Plot total time scaling with feature dimension."""
    
    def filter_and_sort(df):
        if df.empty:
            return pd.DataFrame()
        return df[
            (df["circuit_size"] == circuit_size) &
            (df["batch_size"] == batch_size)
        ].sort_values("feature_dim")
    
    naive_filt = filter_and_sort(naive_df)
    batch_filt = filter_and_sort(batch_df)
    heinsum_filt = filter_and_sort(heinsum_df)

    if not naive_filt.empty:
        ax.plot(naive_filt["feature_dim"], naive_filt["total_time"],
                "^-", label=NAIVE_LABEL, color=colors()[2], linewidth=2,
                markersize=8, markeredgecolor="black", markeredgewidth=1)

    if not batch_filt.empty:
        ax.plot(batch_filt["feature_dim"], batch_filt["total_time"],
                "s-", label=BATCH_LABEL, color=colors()[3], linewidth=2,
                markersize=8, markeredgecolor="black", markeredgewidth=1)

    if not heinsum_filt.empty:
        ax.plot(heinsum_filt["feature_dim"], heinsum_filt["total_time"],
                "o-", label=QTPU_LABEL, color=colors()[0], linewidth=2,
                markersize=8, markeredgecolor="black", markeredgewidth=1)

    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Total Time [s]")
    ax.set_title(r"\textbf{(d) Feature Scaling}" + f"\n{{({circuit_size}q, batch={batch_size})}}", fontsize=8)
    ax.legend(loc="upper left", fontsize=6)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


@bk.pplot
def plot_hybrid_ml_breakdown(
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    circuit_size: int = 8,
    batch_size: int = 100,
    feature_dim: int = 4,
):
    """Main hybrid ML benchmark figure (1 row, 4 columns).
    
    Creates a 1x4 grid showing:
    - (a) Total time comparison by circuit size
    - (b) Time breakdown (prep + quantum + classical)
    - (c) Scaling with batch size
    - (d) Scaling with feature dimension

    Args:
        naive_df: DataFrame with naive benchmark results.
        batch_df: DataFrame with batch benchmark results.
        heinsum_df: DataFrame with HEinsum benchmark results.
        circuit_size: Circuit size for breakdown plot.
        batch_size: Batch size for filtering.
        feature_dim: Feature dimension for filtering.
        
    Returns:
        A BenchKit Plot object.
    """
    fig, axes = plt.subplots(1, 4, figsize=(double_column_width(), 1.8))

    plot_total_time_by_circuit_size(axes[0], naive_df, batch_df, heinsum_df, batch_size, feature_dim)
    plot_time_breakdown(axes[1], naive_df, batch_df, heinsum_df, circuit_size, batch_size, feature_dim)
    plot_batch_scaling(axes[2], naive_df, batch_df, heinsum_df, circuit_size, feature_dim)
    plot_feature_scaling(axes[3], naive_df, batch_df, heinsum_df, circuit_size, batch_size)

    plt.tight_layout()
    return fig


def plot_speedup_heatmap(
    naive_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    feature_dim: int = 4,
    output_path: str = "plots/hybrid_ml/speedup_heatmap.pdf",
):
    """Create a heatmap showing speedup of HEinsum vs Naive across configurations."""
    if naive_df.empty or heinsum_df.empty:
        print("Not enough data for heatmap")
        return None
    
    # Filter for specific feature_dim
    naive_filt = naive_df[naive_df["feature_dim"] == feature_dim]
    heinsum_filt = heinsum_df[heinsum_df["feature_dim"] == feature_dim]
    
    circuit_sizes = sorted(naive_filt["circuit_size"].unique())
    batch_sizes = sorted(naive_filt["batch_size"].unique())
    
    # Create speedup matrix
    speedup_matrix = np.zeros((len(batch_sizes), len(circuit_sizes)))
    
    for i, bs in enumerate(batch_sizes):
        for j, cs in enumerate(circuit_sizes):
            naive_row = naive_filt[(naive_filt["batch_size"] == bs) & (naive_filt["circuit_size"] == cs)]
            heinsum_row = heinsum_filt[(heinsum_filt["batch_size"] == bs) & (heinsum_filt["circuit_size"] == cs)]
            
            if not naive_row.empty and not heinsum_row.empty:
                speedup = naive_row["total_time"].values[0] / heinsum_row["total_time"].values[0]
                speedup_matrix[i, j] = speedup
    
    fig, ax = plt.subplots(figsize=(single_column_width(), 2.0))
    
    im = ax.imshow(speedup_matrix, aspect="auto", cmap="RdYlGn", vmin=0)
    
    ax.set_xticks(np.arange(len(circuit_sizes)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([f"{s}q" for s in circuit_sizes])
    ax.set_yticklabels(batch_sizes)
    
    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Batch Size")
    ax.set_title(rf"\textbf{{Speedup: {QTPU_LABEL} vs {NAIVE_LABEL}}}")
    
    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(circuit_sizes)):
            text = ax.text(j, i, f"{speedup_matrix[i, j]:.1f}x",
                          ha="center", va="center", color="black", fontsize=6)
    
    plt.colorbar(im, ax=ax, label="Speedup")
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    
    return fig


def print_summary_table(
    naive_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    circuit_size: int = 8,
    batch_size: int = 100,
    feature_dim: int = 4,
):
    """Print a summary table comparing the three approaches."""
    print("\n" + "=" * 80)
    print(f"SUMMARY: {circuit_size}q, batch={batch_size}, features={feature_dim}")
    print("=" * 80)
    
    def get_row(df, name):
        if df.empty:
            return None
        filtered = df[
            (df["circuit_size"] == circuit_size) &
            (df["batch_size"] == batch_size) &
            (df["feature_dim"] == feature_dim)
        ]
        return filtered.iloc[0] if not filtered.empty else None
    
    naive = get_row(naive_df, "Naive")
    batch = get_row(batch_df, "Batch")
    heinsum = get_row(heinsum_df, "HEinsum")
    
    print(f"\n{'Metric':<25} {'Naive':>15} {'Batch':>15} {'HEinsum':>15}")
    print("-" * 70)
    
    if naive is not None:
        n_prep, n_quantum, n_class = naive["preparation_time"], naive["quantum_time"], naive.get("classical_time", 0)
        n_total = naive["total_time"]
    else:
        n_prep = n_quantum = n_class = n_total = float('nan')
    
    if batch is not None:
        b_prep, b_quantum, b_class = batch["preparation_time"], batch["quantum_time"], batch.get("classical_time", 0)
        b_total = batch["total_time"]
    else:
        b_prep = b_quantum = b_class = b_total = float('nan')
    
    if heinsum is not None:
        h_prep, h_quantum, h_class = heinsum["preparation_time"], heinsum["quantum_time"], heinsum.get("classical_time", 0)
        h_total = heinsum["total_time"]
    else:
        h_prep = h_quantum = h_class = h_total = float('nan')
    
    print(f"{'Prep Time (s)':<25} {n_prep:>15.4f} {b_prep:>15.4f} {h_prep:>15.4f}")
    print(f"{'Quantum Time (s)':<25} {n_quantum:>15.4f} {b_quantum:>15.4f} {h_quantum:>15.4f}")
    print(f"{'Classical Time (s)':<25} {n_class:>15.4f} {b_class:>15.4f} {h_class:>15.4f}")
    print(f"{'Total Time (s)':<25} {n_total:>15.4f} {b_total:>15.4f} {h_total:>15.4f}")
    
    if heinsum is not None and naive is not None:
        print(f"\nSpeedups (vs Naive):")
        print(f"  Prep: {n_prep/h_prep:.1f}x")
        print(f"  Total: {n_total/h_total:.1f}x")
    
    if heinsum is not None and batch is not None:
        print(f"\nSpeedups (vs Batch):")
        print(f"  Prep: {b_prep/h_prep:.1f}x")
        print(f"  Total: {b_total/h_total:.1f}x")


if __name__ == "__main__":
    # Disable TeX for local testing
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    register_style("heinsum", PlotStyle(color=colors()[0], hatch="///"))
    register_style("batch", PlotStyle(color=colors()[3], hatch="\\\\\\"))
    register_style("naive", PlotStyle(color=colors()[2], hatch="xxx"))

    naive_path = "logs/hybrid_ml/naive_breakdown.jsonl"
    batch_path = "logs/hybrid_ml/batch_breakdown.jsonl"
    heinsum_path = "logs/hybrid_ml/heinsum_breakdown.jsonl"

    naive_df, batch_df, heinsum_df = load_and_prepare_data(naive_path, batch_path, heinsum_path)

    print("Data loaded:")
    print(f"  Naive: {len(naive_df)} rows")
    print(f"  Batch: {len(batch_df)} rows")
    print(f"  HEinsum: {len(heinsum_df)} rows")

    if not naive_df.empty:
        print("\nNaive configurations:")
        print(f"  Circuit sizes: {sorted(naive_df['circuit_size'].unique())}")
        print(f"  Batch sizes: {sorted(naive_df['batch_size'].unique())}")
        print(f"  Feature dims: {sorted(naive_df['feature_dim'].unique())}")

    # Print summary table for a specific configuration
    print_summary_table(naive_df, batch_df, heinsum_df, circuit_size=8, batch_size=100, feature_dim=4)

    # Generate main figure
    if not naive_df.empty or not batch_df.empty or not heinsum_df.empty:
        Path("plots/hybrid_ml").mkdir(parents=True, exist_ok=True)
        
        fig = plot_hybrid_ml_breakdown(
            naive_df, batch_df, heinsum_df,
            circuit_size=8, batch_size=100, feature_dim=4
        )
        plt.savefig("plots/hybrid_ml/breakdown.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("plots/hybrid_ml/breakdown.png", dpi=300, bbox_inches="tight")
        print("\nSaved plots/hybrid_ml/breakdown.pdf")

        # Speedup heatmap
        if not naive_df.empty and not heinsum_df.empty:
            plot_speedup_heatmap(naive_df, heinsum_df, feature_dim=4)

        plt.show()
    else:
        print("\nNo data found. Run the benchmarks first:")
        print("  python evaluation/use_cases/hybrid_ml/run.py all")
