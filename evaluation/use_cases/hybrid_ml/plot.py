"""
Plot Hybrid ML Inference Benchmark Results
==========================================

Creates publication-quality plots showing:
1. Total time comparison by circuit size
2. Time breakdown (Prep + Quantum + Classical) at 100q
3. Batch size scaling at 100 qubits
4. Code size scaling with circuit size
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

# Consistent hatches across all plots
HATCHES = {"batch": "\\\\\\", "heinsum": "///"}


def load_and_prepare_data(
    batch_path: str, heinsum_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load JSONL logs and convert to DataFrames."""
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
            "result.total_code_lines": "total_code_lines",
        }
        return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    batch_df = rename_cols(batch_df)
    heinsum_df = rename_cols(heinsum_df)

    # Convert memory from bytes to KB
    for df in [batch_df, heinsum_df]:
        if not df.empty and "peak_memory" in df.columns:
            df["peak_memory_kb"] = df["peak_memory"] / 1024

    return batch_df, heinsum_df


def plot_preparation_speedup(
    ax,
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    batch_size: int = 100,
    feature_dim: int = 2,
):
    """Plot preparation time speedup - this is where qTPU shines."""
    
    def filter_df(df, name):
        if df.empty:
            return pd.DataFrame()
        filtered = df[
            (df["batch_size"] == batch_size) & 
            (df["feature_dim"] == feature_dim)
        ].sort_values("circuit_size")
        return filtered
    
    batch_filt = filter_df(batch_df, "batch")
    heinsum_filt = filter_df(heinsum_df, "heinsum")

    circuit_sizes = []
    for df in [batch_filt, heinsum_filt]:
        if not df.empty:
            circuit_sizes = sorted(df["circuit_size"].unique())
            break

    if len(circuit_sizes) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(circuit_sizes))
    width = 0.35

    # Extract prep times
    batch_prep = [batch_filt[batch_filt["circuit_size"] == s]["preparation_time"].values[0]
                  if s in batch_filt["circuit_size"].values else 0 for s in circuit_sizes]
    heinsum_prep = [heinsum_filt[heinsum_filt["circuit_size"] == s]["preparation_time"].values[0]
                    if s in heinsum_filt["circuit_size"].values else 0 for s in circuit_sizes]

    ax.bar(x - width/2, heinsum_prep, width, label=QTPU_LABEL,
           color=colors()[0], edgecolor="black", linewidth=0.5, hatch=HATCHES["heinsum"])
    ax.bar(x + width/2, batch_prep, width, label=BATCH_LABEL,
           color=colors()[5], edgecolor="black", linewidth=0.5, hatch=HATCHES["batch"])

    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Compilation Time [s]")
    ax.set_title(r"\textbf{(a) Compilation by Circuit Size}" + "\n" + r"\textit{lower is better $\downarrow$}", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in circuit_sizes])
    ax.legend(loc="upper left", fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")


def plot_compilation_scalability(
    ax,
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    circuit_size: int = 100,
    feature_dim: int = 2,
):
    """Show how compilation/prep time scales with batch size."""
    
    def filter_data(df):
        if df.empty:
            return pd.DataFrame()
        filtered = df[
            (df["circuit_size"] == circuit_size) &
            (df["feature_dim"] == feature_dim)
        ].sort_values("batch_size")
        return filtered
    
    batch_filt = filter_data(batch_df)
    heinsum_filt = filter_data(heinsum_df)

    batch_sizes = []
    for df in [batch_filt, heinsum_filt]:
        if not df.empty:
            batch_sizes = sorted([bs for bs in df["batch_size"].unique() if bs != 10])
            break

    if len(batch_sizes) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    batch_prep_times = []
    heinsum_prep_times = []
    
    for bs in batch_sizes:
        if bs in batch_filt["batch_size"].values:
            row = batch_filt[batch_filt["batch_size"] == bs].iloc[0]
            batch_prep_times.append(row["preparation_time"])
        else:
            batch_prep_times.append(None)
            
        if bs in heinsum_filt["batch_size"].values:
            row = heinsum_filt[heinsum_filt["batch_size"] == bs].iloc[0]
            heinsum_prep_times.append(row["preparation_time"])
        else:
            heinsum_prep_times.append(None)

    x = np.arange(len(batch_sizes))
    width = 0.35

    # Bar plot with qTPU on left
    ax.bar(x - width/2, heinsum_prep_times, width, label=QTPU_LABEL,
           color=colors()[0], edgecolor="black", linewidth=0.5, hatch=HATCHES["heinsum"])
    ax.bar(x + width/2, batch_prep_times, width, label=BATCH_LABEL,
           color=colors()[5], edgecolor="black", linewidth=0.5, hatch=HATCHES["batch"])

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Compilation Time [s]")
    ax.set_title(r"\textbf{(b) Compilation by Batch Size}" + "\n" + r"\textit{lower is better $\downarrow$}", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bs}" for bs in batch_sizes])
    ax.legend(loc="upper left", fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")


def plot_code_reduction(
    ax,
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    batch_size: int = 100,
):
    """Plot code reduction - show both absolute and reduction factor."""
    
    def filter_and_sort(df):
        if df.empty:
            return pd.DataFrame()
        filtered = df[df["batch_size"] == batch_size]
        if filtered.empty:
            return pd.DataFrame()
        return filtered.groupby("circuit_size").first().reset_index().sort_values("circuit_size")
    
    batch_filt = filter_and_sort(batch_df)
    heinsum_filt = filter_and_sort(heinsum_df)

    # Get circuit sizes
    circuit_sizes = []
    if not batch_filt.empty:
        circuit_sizes = sorted(batch_filt["circuit_size"].unique())
    elif not heinsum_filt.empty:
        circuit_sizes = sorted(heinsum_filt["circuit_size"].unique())
    
    if not circuit_sizes:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(circuit_sizes))
    width = 0.35

    batch_code = [batch_filt[batch_filt["circuit_size"] == s]["total_code_lines"].values[0]
                   if s in batch_filt["circuit_size"].values and "total_code_lines" in batch_filt.columns else 0 
                   for s in circuit_sizes]
    heinsum_code = [heinsum_filt[heinsum_filt["circuit_size"] == s]["total_code_lines"].values[0]
                     if s in heinsum_filt["circuit_size"].values and "total_code_lines" in heinsum_filt.columns else 0 
                     for s in circuit_sizes]

    ax.bar(x - width/2, heinsum_code, width, label=QTPU_LABEL,
           color=colors()[0], edgecolor="black", linewidth=0.5, hatch=HATCHES["heinsum"])
    ax.bar(x + width/2, batch_code, width, label=BATCH_LABEL,
           color=colors()[5], edgecolor="black", linewidth=0.5, hatch=HATCHES["batch"])

    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Generated Code Lines")
    ax.set_title(r"\textbf{(c) Code Reduction}" + "\n" + r"\textit{fewer is better $\downarrow$}", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in circuit_sizes])
    ax.legend(loc="upper left", fontsize=6)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")


@bk.pplot
def plot_hybrid_ml_benchmark(
    batch_df: pd.DataFrame,
    heinsum_df: pd.DataFrame,
    batch_size: int = 100,
    feature_dim: int = 2,
):
    """Main hybrid ML benchmark figure highlighting key benefits.
    
    Creates a 1x3 grid showing:
    - (a) Preparation time speedup by circuit size
    - (b) Code reduction by circuit size
    - (c) Compilation time scalability with batch size

    Args:
        batch_df: DataFrame with batch benchmark results.
        heinsum_df: DataFrame with HEinsum benchmark results.
        batch_size: Batch size for filtering.
        feature_dim: Feature dimension for filtering.
        
    Returns:
        A BenchKit Plot object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.8))

    plot_preparation_speedup(axes[0], batch_df, heinsum_df, batch_size, feature_dim)
    plot_compilation_scalability(axes[1], batch_df, heinsum_df, 100, feature_dim)
    plot_code_reduction(axes[2], batch_df, heinsum_df, batch_size)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Disable TeX for local testing
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    register_style("heinsum", PlotStyle(color=colors()[0], hatch="///"))
    register_style("batch", PlotStyle(color=colors()[5], hatch="\\\\\\\\\\\\\\"))

    batch_path = "logs/hybrid_ml/batch_breakdown.jsonl"
    heinsum_path = "logs/hybrid_ml/heinsum_breakdown.jsonl"

    batch_df, heinsum_df = load_and_prepare_data(batch_path, heinsum_path)

    # Print comprehensive metrics across all configurations
    if not batch_df.empty and not heinsum_df.empty:
        print("\n" + "="*70)
        print("HYBRID ML BENCHMARK SUMMARY")
        print("="*70)
        
        # Calculate metrics across all matching pairs
        prep_speedups = []
        code_reductions = []
        total_speedups = []
        
        for _, batch_row in batch_df.iterrows():
            heinsum_match = heinsum_df[
                (heinsum_df['circuit_size'] == batch_row['circuit_size']) & 
                (heinsum_df['batch_size'] == batch_row['batch_size']) & 
                (heinsum_df['feature_dim'] == batch_row['feature_dim'])
            ]
            
            if not heinsum_match.empty:
                heinsum_row = heinsum_match.iloc[0]
                
                if heinsum_row['preparation_time'] > 0:
                    prep_speedups.append(batch_row['preparation_time'] / heinsum_row['preparation_time'])
                
                if heinsum_row['total_code_lines'] > 0:
                    code_reductions.append(batch_row['total_code_lines'] / heinsum_row['total_code_lines'])
                
                # Calculate correct total time: prep + quantum
                batch_total = batch_row['preparation_time'] + batch_row['quantum_time']
                heinsum_total = heinsum_row['preparation_time'] + heinsum_row['quantum_time']
                if heinsum_total > 0:
                    total_speedups.append(batch_total / heinsum_total)
        
        # Print summary statistics
        print("\nCompilation Time Speedup:")
        print(f"  Average: {np.mean(prep_speedups):.2f}x")
        print(f"  Median:  {np.median(prep_speedups):.2f}x")
        print(f"  Range:   {np.min(prep_speedups):.2f}x - {np.max(prep_speedups):.2f}x")
        
        print("\nCode Size Reduction:")
        print(f"  Average: {np.mean(code_reductions):.1f}x")
        print(f"  Median:  {np.median(code_reductions):.1f}x")
        print(f"  Up to:   {np.max(code_reductions):.1f}x")
        
        print("\nEnd-to-End Speedup:")
        print(f"  Average: {np.mean(total_speedups):.2f}x")
        print(f"  Median:  {np.median(total_speedups):.2f}x")
        print(f"  Up to:   {np.max(total_speedups):.2f}x")
        
        print("="*70 + "\n")

    # Generate main figure
    if not batch_df.empty or not heinsum_df.empty:
        Path("plots/hybrid_ml").mkdir(parents=True, exist_ok=True)
        
        # Main benchmark figure
        fig = plot_hybrid_ml_benchmark(
            batch_df, heinsum_df,
            batch_size=50, feature_dim=2
        )
        plt.savefig("plots/hybrid_ml/benchmark.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("plots/hybrid_ml/benchmark.png", dpi=300, bbox_inches="tight")
        print("\nSaved plots/hybrid_ml/benchmark.pdf")

        plt.show()
    else:
        print("\nNo data found. Run the benchmarks first:")
        print("  python evaluation/use_cases/hybrid_ml/run.py all")
