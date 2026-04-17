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

ESTIMATED_LABEL = r"\textsc{Estimated}"
ACTUAL_LABEL = r"\textsc{Actual}"


def _sizes(df: pd.DataFrame) -> list[int]:
    return sorted(int(s) for s in df["config.circuit_size"].unique())


def plot_qpu_time_validation(ax, df: pd.DataFrame):
    """(a) Estimated vs. actual QPU time on IBM Marrakesh."""
    sizes = _sizes(df)
    est_vals, act_vals = [], []
    for s in sizes:
        row = df[df["config.circuit_size"] == s]
        if row.empty:
            continue
        est_vals.append(row.iloc[0]["result.estimated_qpu_time"])
        act_vals.append(row.iloc[0]["result.actual_qpu_time"])

    x = np.arange(len(sizes))
    width = 0.35

    ax.bar(
        x - width / 2,
        est_vals,
        width,
        label=ESTIMATED_LABEL,
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )
    ax.bar(
        x + width / 2,
        act_vals,
        width,
        label=ACTUAL_LABEL,
        color=colors()[1],
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )

    ax.set_xlabel("Circuit Size ")
    ax.set_ylabel("QPU Time [s]")
    ax.set_title(r"\textbf{(a) QPU Time Validation}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in sizes])


def plot_fidelity(ax, df: pd.DataFrame):
    """(b) Hardware fidelity vs. circuit size (higher is better)."""
    sizes = _sizes(df)
    fids = []
    for s in sizes:
        row = df[df["config.circuit_size"] == s]
        if row.empty:
            continue
        fids.append(row.iloc[0]["result.mean_fidelity"])

    x = np.arange(len(sizes))

    ax.bar(
        x,
        fids,
        0.5,
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )

    ax.set_xlabel("Circuit Size ")
    ax.set_ylabel("Fidelity\n(higher is better)")
    ax.set_title(r"\textbf{(b) Hardware Fidelity}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in sizes])


@bk.pplot
def plot_hardware(df: pd.DataFrame):
    """Real IBM Marrakesh hardware benchmark: time validation + fidelity."""
    fig, axes = plt.subplots(1, 2, figsize=(double_column_width() * 2 / 3, 1.3))

    plot_qpu_time_validation(axes[0], df)
    plot_fidelity(axes[1], df)

    return fig


if __name__ == "__main__":
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    register_style("estimated", PlotStyle(color=colors()[0], hatch="//"))
    register_style("actual", PlotStyle(color=colors()[1], hatch="\\\\"))

    data = bk.load_log("logs/hardware/qnn.jsonl")
    df = data if isinstance(data, pd.DataFrame) else pd.json_normalize(data)

    if df.empty:
        print("No hardware data found")
        exit(1)

    print(f"Hardware: {len(df)} rows, sizes={_sizes(df)}")

    fig = plot_hardware(df)
    plt.tight_layout()
    plt.savefig("plots/hardware.pdf", bbox_inches="tight")
    plt.show()
