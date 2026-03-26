"""Plot error mitigation benchmark results (Fig 14)."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils import (
    colors,
    single_column_width,
    load_results,
    setup_paper_style,
)

QTPU_LABEL = "qTPU"
MITIQ_LABEL = "Mitiq"

QTPU_COLOR = colors()[0]
MITIQ_COLOR = colors()[3]


def plot_compile_time(ax, qtpu_df: pd.DataFrame, mitiq_df: pd.DataFrame):
    available_samples = sorted(mitiq_df["config.num_samples"].unique())
    sample_sizes = [s for s in [100, 1000, 10000] if s in available_samples]
    labels = {100: "100", 1000: "1k", 10000: "10k"}

    qtpu_avg = qtpu_df["result.compilation_time"].mean()
    mitiq_times = [
        mitiq_df[mitiq_df["config.num_samples"] == s]["result.compilation_time"].mean()
        for s in sample_sizes
    ]

    x = np.arange(len(sample_sizes))
    w = 0.32

    ax.bar(x - w / 2, [qtpu_avg] * len(sample_sizes), w,
           label=QTPU_LABEL, color=QTPU_COLOR, edgecolor="black",
           linewidth=0.4, hatch="//")
    ax.bar(x + w / 2, mitiq_times, w,
           label=MITIQ_LABEL, color=MITIQ_COLOR, edgecolor="black",
           linewidth=0.4, hatch="\\\\")

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Compile Time [s]\n(lower is better)")
    ax.set_title("(a) Compile Time", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in sample_sizes])


def plot_code_lines(ax, qtpu_df: pd.DataFrame, mitiq_df: pd.DataFrame):
    available_samples = sorted(mitiq_df["config.num_samples"].unique())
    sample_sizes = [s for s in [100, 1000, 10000] if s in available_samples]
    labels = {100: "100", 1000: "1k", 10000: "10k"}

    qtpu_avg = qtpu_df["result.total_code_lines"].mean()
    mitiq_lines = [
        mitiq_df[mitiq_df["config.num_samples"] == s]["result.total_code_lines"].mean()
        for s in sample_sizes
    ]

    x = np.arange(len(sample_sizes))
    w = 0.32

    ax.bar(x - w / 2, [qtpu_avg] * len(sample_sizes), w,
           label=QTPU_LABEL, color=QTPU_COLOR, edgecolor="black",
           linewidth=0.4, hatch="//")
    ax.bar(x + w / 2, mitiq_lines, w,
           label=MITIQ_LABEL, color=MITIQ_COLOR, edgecolor="black",
           linewidth=0.4, hatch="\\\\")

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Code Size [LoC]\n(fewer is better)")
    ax.set_title("(b) Code Size", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in sample_sizes])


def plot_error_mitigation_comparison(qtpu_df: pd.DataFrame, mitiq_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(single_column_width(), 1.5))
    plot_compile_time(axes[0], qtpu_df, mitiq_df)
    plot_code_lines(axes[1], qtpu_df, mitiq_df)
    fig.tight_layout(w_pad=1.5)
    return fig


if __name__ == "__main__":
    matplotlib.use("Agg")
    setup_paper_style()

    qtpu_df = load_results("logs/error_mitigation/qtpu_breakdown.jsonl")
    mitiq_df = load_results("logs/error_mitigation/mitiq_breakdown.jsonl")

    qtpu_df = qtpu_df[qtpu_df["config.mitigation"].notna()]
    if qtpu_df.empty:
        print("No QTPU data found")
        exit(1)

    # Summary
    qtpu_avg_time = qtpu_df["result.compilation_time"].mean()
    qtpu_avg_lines = qtpu_df["result.total_code_lines"].mean()
    print(f"QTPU: {len(qtpu_df)} rows, Mitiq: {len(mitiq_df)} rows")
    print(f"QTPU: {qtpu_avg_time*1000:.2f}ms, {qtpu_avg_lines:.0f} LoC")

    if not mitiq_df.empty:
        for s in sorted(mitiq_df["config.num_samples"].unique()):
            md = mitiq_df[mitiq_df["config.num_samples"] == s]
            t = md["result.compilation_time"].mean()
            c = md["result.total_code_lines"].mean()
            print(f"Mitiq {int(s):>5}: {t*1000:>8.1f}ms ({t/qtpu_avg_time:>5.0f}x), "
                  f"{c:>10,.0f} LoC ({c/qtpu_avg_lines:>5.0f}x)")

    from pathlib import Path
    Path("plots").mkdir(parents=True, exist_ok=True)

    fig = plot_error_mitigation_comparison(qtpu_df, mitiq_df)
    fig.savefig("plots/error_mitigation.pdf")
    print("Saved plots/error_mitigation.pdf")
