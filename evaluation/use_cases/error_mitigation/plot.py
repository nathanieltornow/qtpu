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
    sample_sizes = [100, 1000, 10000]
    sample_labels = ["100", "1k", "10k"]
    
    # Average QTPU time across all circuit sizes and mitigations
    qtpu_avg_time = qtpu_df["result.generation_time"].mean()
    
    # Average Mitiq time for each sample size
    mitiq_times = []
    for samples in sample_sizes:
        mitiq_data = mitiq_df[mitiq_df["config.num_samples"] == samples]
        mitiq_times.append(mitiq_data["result.generation_time"].mean())
    
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
    
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Compile Time [s]")
    ax.set_title(r"\textbf{(a) Compile Time}" + "\n" + r"{(lower is better $\downarrow$)}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_labels)


def plot_memory_usage(ax, qtpu_df: pd.DataFrame, mitiq_df: pd.DataFrame):
    """Plot 2: Bar plot comparing QTPU vs Mitiq-style memory usage."""
    sample_sizes = [100, 1000, 10000]
    sample_labels = ["100", "1k", "10k"]
    
    # Average QTPU memory across all circuit sizes and mitigations (convert to KB)
    qtpu_avg_mem = qtpu_df["result.generation_memory"].mean() / 1024
    
    # Average Mitiq memory for each sample size (convert to KB)
    mitiq_mem = []
    for samples in sample_sizes:
        mitiq_data = mitiq_df[mitiq_df["config.num_samples"] == samples]
        mitiq_mem.append(mitiq_data["result.generation_memory"].mean() / 1024)
    
    x = np.arange(len(sample_sizes))
    width = 0.35
    
    # QTPU bars
    ax.bar(
        x - width / 2,
        [qtpu_avg_mem] * len(sample_sizes),
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
        mitiq_mem,
        width,
        label=MITIQ_LABEL,
        color=colors()[3],
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )
    
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Memory [KB]")
    ax.set_title(r"\textbf{(b) Memory}" + "\n" + r"{(lower is better $\downarrow$)}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_labels)


@bk.pplot
def plot_error_mitigation_comparison(qtpu_df: pd.DataFrame, mitiq_df: pd.DataFrame):
    """Plot error mitigation comparison between QTPU and Mitiq-style.

    Args:
        qtpu_df: DataFrame with QTPU benchmark results.
        mitiq_df: DataFrame with Mitiq-style benchmark results.
    Returns:
        A BenchKit Plot object comparing the two approaches.
    """
    fig, axes = plt.subplots(1, 2, figsize=(single_column_width(), 1.6))

    plot_compile_time(axes[0], qtpu_df, mitiq_df)
    plot_memory_usage(axes[1], qtpu_df, mitiq_df)

    return fig


if __name__ == "__main__":
    # Disable TeX for local testing
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    register_style("qtpu", PlotStyle(color=colors()[0], hatch="//"))
    register_style("mitiq", PlotStyle(color=colors()[2], hatch="o"))

    # Load data
    qtpu_df = bk.load_log("logs/error_mitigation/qtpu.jsonl")
    mitiq_df = bk.load_log("logs/error_mitigation/mitiq.jsonl")

    if qtpu_df.empty:
        print("No QTPU data found")
        exit(1)

    print(f"QTPU: {len(qtpu_df)} rows, Mitiq: {len(mitiq_df)} rows")
    print(f"QTPU avg time: {qtpu_df['result.generation_time'].mean()*1000:.2f}ms")
    print(f"Mitiq sample sizes: {sorted(mitiq_df['config.num_samples'].unique())}")

    fig = plot_error_mitigation_comparison(qtpu_df, mitiq_df)
    plt.tight_layout()
    plt.savefig("plots/error_mitigation.pdf", bbox_inches="tight")
    plt.show()
