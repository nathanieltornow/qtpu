"""Plot error mitigation benchmark results."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchkit.plot.config import (
    PlotStyle,
    colors,
    single_column_width,
    register_style,
)

import benchkit as bk

QTPU_LABEL = r"\textsc{qTPU}"
MITIQ_LABEL = r"\textsc{Mitiq}"


def plot_compile_time(ax, qtpu_df: pd.DataFrame, mitiq_df: pd.DataFrame):
    """Plot 1: Bar plot comparing QTPU vs Mitiq-style compile time."""
    # Get available sample sizes from Mitiq data
    available_samples = sorted(mitiq_df["config.num_samples"].unique())
    sample_sizes = [s for s in [100, 1000, 10000] if s in available_samples]
    sample_labels = {100: "100", 1000: "1k", 10000: "10k"}
    labels = [sample_labels[s] for s in sample_sizes]
    
    if not sample_sizes:
        print("No matching sample sizes found")
        return
    
    # Average QTPU time across all mitigations
    qtpu_avg_time = qtpu_df["result.compilation_time"].mean()
    
    # Average Mitiq time for each sample size
    mitiq_times = []
    for samples in sample_sizes:
        mitiq_data = mitiq_df[mitiq_df["config.num_samples"] == samples]
        mitiq_times.append(mitiq_data["result.compilation_time"].mean())
    
    x = np.arange(len(sample_sizes))
    width = 0.35
    
    # QTPU bars (same value for all x positions since it doesn't depend on samples)
    ax.bar(
        x - width / 2,
        [qtpu_avg_time] * len(sample_sizes),
        width,
        label=QTPU_LABEL,
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )
    
    # Mitiq bars
    ax.bar(
        x + width / 2,
        mitiq_times,
        width,
        label=MITIQ_LABEL,
        color=colors()[3],
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )
    
    ax.set_xlabel("Number of Samples", labelpad=2)
    ax.set_ylabel("Compile Time [s]\n(lower is better)")
    ax.set_title(r"\textbf{(a) Compile Time}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


def plot_code_lines(ax, qtpu_df: pd.DataFrame, mitiq_df: pd.DataFrame):
    """Plot 2: Bar plot comparing QTPU vs Mitiq-style code lines."""
    # Get available sample sizes from Mitiq data
    available_samples = sorted(mitiq_df["config.num_samples"].unique())
    sample_sizes = [s for s in [100, 1000, 10000] if s in available_samples]
    sample_labels = {100: "100", 1000: "1k", 10000: "10k"}
    labels = [sample_labels[s] for s in sample_sizes]
    
    if not sample_sizes:
        print("No matching sample sizes found")
        return
    
    # Average QTPU code lines across all mitigations
    qtpu_avg_lines = qtpu_df["result.total_code_lines"].mean()
    
    # Average Mitiq code lines for each sample size
    mitiq_lines = []
    for samples in sample_sizes:
        mitiq_data = mitiq_df[mitiq_df["config.num_samples"] == samples]
        mitiq_lines.append(mitiq_data["result.total_code_lines"].mean())
    
    x = np.arange(len(sample_sizes))
    width = 0.35
    
    # QTPU bars
    ax.bar(
        x - width / 2,
        [qtpu_avg_lines] * len(sample_sizes),
        width,
        label=QTPU_LABEL,
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )
    
    # Mitiq bars
    ax.bar(
        x + width / 2,
        mitiq_lines,
        width,
        label=MITIQ_LABEL,
        color=colors()[3],
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )
    
    ax.set_xlabel("Number of Samples", labelpad=2)
    ax.set_ylabel("Code Size [LoC]\n(fewer is better)")
    ax.set_title(r"\textbf{(b) Code Size}")
    ax.set_yscale("log")
    # ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


@bk.pplot
def plot_error_mitigation_comparison(qtpu_df: pd.DataFrame, mitiq_df: pd.DataFrame):
    """Plot error mitigation comparison between QTPU and Mitiq-style.

    Args:
        qtpu_df: DataFrame with QTPU benchmark results.
        mitiq_df: DataFrame with Mitiq-style benchmark results.
    Returns:
        A BenchKit Plot object comparing the two approaches.
    """
    fig, axes = plt.subplots(1, 2, figsize=(single_column_width(), 1.3))

    plot_compile_time(axes[0], qtpu_df, mitiq_df)
    plot_code_lines(axes[1], qtpu_df, mitiq_df)

    return fig


if __name__ == "__main__":
    # Disable TeX for local testing
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    register_style("qtpu", PlotStyle(color=colors()[0], hatch="//"))
    register_style("mitiq", PlotStyle(color=colors()[2], hatch="o"))

    # Load data
    qtpu_df = bk.load_log("logs/error_mitigation/qtpu_breakdown.jsonl")
    mitiq_df = bk.load_log("logs/error_mitigation/mitiq_breakdown.jsonl")

    # Filter to only new data with 'mitigation' config
    qtpu_df = qtpu_df[qtpu_df["config.mitigation"].notna()]
    
    if qtpu_df.empty:
        print("No QTPU data found")
        exit(1)

    print(f"QTPU: {len(qtpu_df)} rows, Mitiq: {len(mitiq_df)} rows")
    print(f"QTPU avg time: {qtpu_df['result.compilation_time'].mean()*1000:.2f}ms")
    print(f"QTPU avg code lines: {qtpu_df['result.total_code_lines'].mean():.0f}")
    if not mitiq_df.empty:
        print(f"Mitiq sample sizes: {sorted(mitiq_df['config.num_samples'].unique())}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Error Mitigation Benchmark Summary")
    print("="*80)
    
    # QTPU summary
    qtpu_avg_time = qtpu_df['result.compilation_time'].mean()
    qtpu_avg_lines = qtpu_df['result.total_code_lines'].mean()
    print(f"\nQTPU (single kernel for all combinations):")
    print(f"  Compilation time: {qtpu_avg_time*1000:.2f}ms")
    print(f"  Code lines: {qtpu_avg_lines:.0f}")
    
    # Mitiq summary for each sample size
    if not mitiq_df.empty:
        sample_sizes = sorted([s for s in [100, 1000, 10000] if s in mitiq_df['config.num_samples'].unique()])
        print(f"\nMitiq (separate circuits per sample):")
        for samples in sample_sizes:
            mitiq_data = mitiq_df[mitiq_df['config.num_samples'] == samples]
            avg_time = mitiq_data['result.compilation_time'].mean()
            avg_lines = mitiq_data['result.total_code_lines'].mean()
            speedup = avg_time / qtpu_avg_time
            code_ratio = avg_lines / qtpu_avg_lines
            print(f"  {samples:>5} samples: time={avg_time*1000:>8.2f}ms ({speedup:>6.1f}x slower), "
                  f"code={avg_lines:>12,.0f} ({code_ratio:>6.1f}x more)")
    
    print("="*80 + "\n")

    fig = plot_error_mitigation_comparison(qtpu_df, mitiq_df)
    plt.tight_layout()
    plt.savefig("plots/error_mitigation.pdf", bbox_inches="tight")
    plt.show()
