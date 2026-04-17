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
BASELINE_LABEL = r"\textsc{Baseline}"


def _sizes(df: pd.DataFrame) -> list[int]:
    return sorted(int(s) for s in df["config.circuit_size"].unique())


def _lookup(df: pd.DataFrame, size: int, key: str, default: float = 0.0) -> float:
    row = df[df["config.circuit_size"] == size]
    if row.empty or key not in row.columns:
        return default
    val = row.iloc[0][key]
    if pd.isna(val):
        return default
    return float(val)


def _wall_time(row_df: pd.DataFrame, size: int) -> float:
    """End-to-end wall time = compile + estimated quantum + classical."""
    return (
        _lookup(row_df, size, "result.compile_time")
        + _lookup(row_df, size, "result.quantum_time")
        + _lookup(row_df, size, "result.classical_time")
    )


def plot_end_to_end_time(ax, qtpu_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """(a) End-to-end wall time (lower is better, log scale).

    The first metric from the revision commitment: compile + estimated quantum
    (FakeMarrakesh + ASAP scheduling) + classical post-processing, summed.
    """
    sizes = _sizes(qtpu_df)
    qtpu_vals = [_wall_time(qtpu_df, s) for s in sizes]
    base_vals = [_wall_time(baseline_df, s) for s in sizes]

    x = np.arange(len(sizes))
    width = 0.35

    ax.bar(x - width / 2, qtpu_vals, width,
           label=QTPU_LABEL, color=colors()[0],
           edgecolor="black", linewidth=1, hatch="//")
    ax.bar(x + width / 2, base_vals, width,
           label=BASELINE_LABEL, color=colors()[1],
           edgecolor="black", linewidth=1, hatch="\\\\")

    ax.set_xlabel("Circuit Size ")
    ax.set_ylabel("End-to-End Time [s]\n(lower is better)")
    ax.set_title(r"\textbf{(a) End-to-End Wall Time}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in sizes])


def plot_generated_circuits(ax, qtpu_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """(b) Number of generated circuits (lower is better, log scale)."""
    sizes = _sizes(qtpu_df)
    qtpu_vals = [_lookup(qtpu_df, s, "result.num_circuits") for s in sizes]
    base_vals = [_lookup(baseline_df, s, "result.num_circuits") for s in sizes]

    x = np.arange(len(sizes))
    width = 0.35

    ax.bar(x - width / 2, qtpu_vals, width,
           label=QTPU_LABEL, color=colors()[0],
           edgecolor="black", linewidth=1, hatch="//")
    ax.bar(x + width / 2, base_vals, width,
           label=BASELINE_LABEL, color=colors()[1],
           edgecolor="black", linewidth=1, hatch="\\\\")

    ax.set_xlabel("Circuit Size ")
    ax.set_ylabel("Circuits Generated\n(lower is better)")
    ax.set_title(r"\textbf{(b) Generated Circuits}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in sizes])


def plot_code_size(ax, qtpu_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """(c) Lines of generated code (lower is better, log scale)."""
    sizes = _sizes(qtpu_df)
    qtpu_vals = [_lookup(qtpu_df, s, "result.total_code_lines") for s in sizes]
    base_vals = [_lookup(baseline_df, s, "result.total_code_lines") for s in sizes]

    x = np.arange(len(sizes))
    width = 0.35

    ax.bar(x - width / 2, qtpu_vals, width,
           label=QTPU_LABEL, color=colors()[0],
           edgecolor="black", linewidth=1, hatch="//")
    ax.bar(x + width / 2, base_vals, width,
           label=BASELINE_LABEL, color=colors()[1],
           edgecolor="black", linewidth=1, hatch="\\\\")

    ax.set_xlabel("Circuit Size ")
    ax.set_ylabel("Lines of Code\n(lower is better)")
    ax.set_title(r"\textbf{(c) Generated Code Size}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in sizes])


@bk.pplot
def plot_end_to_end(qtpu_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """End-to-end composability benchmark: qTPU vs. baseline pipeline.

    The three metrics committed in revision_plan.md §Condition 1:
      (a) End-to-end wall time.
      (b) Number of generated circuits.
      (c) Lines of generated code.
    """
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.3))

    plot_end_to_end_time(axes[0], qtpu_df, baseline_df)
    plot_generated_circuits(axes[1], qtpu_df, baseline_df)
    plot_code_size(axes[2], qtpu_df, baseline_df)

    return fig


if __name__ == "__main__":
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    register_style("qtpu", PlotStyle(color=colors()[0], hatch="//"))
    register_style("baseline", PlotStyle(color=colors()[1], hatch="\\\\"))

    qtpu_data = bk.load_log("logs/end_to_end/qtpu.jsonl")
    baseline_data = bk.load_log("logs/end_to_end/baseline.jsonl")

    qtpu_df = qtpu_data if isinstance(qtpu_data, pd.DataFrame) else pd.json_normalize(qtpu_data)
    baseline_df = (
        baseline_data if isinstance(baseline_data, pd.DataFrame)
        else (pd.json_normalize(baseline_data) if baseline_data else pd.DataFrame())
    )

    if qtpu_df.empty or baseline_df.empty:
        print("Missing qTPU or baseline data")
        exit(1)

    print(f"qTPU: {len(qtpu_df)} rows, Baseline: {len(baseline_df)} rows")
    print(f"Sizes: {_sizes(qtpu_df)}")

    fig = plot_end_to_end(qtpu_df, baseline_df)
    plt.tight_layout()
    plt.savefig("plots/end_to_end.pdf", bbox_inches="tight")
    plt.show()
