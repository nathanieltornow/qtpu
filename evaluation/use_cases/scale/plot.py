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
QAC_LABEL = r"\textsc{QAC}"
TIMEOUT_VALUE = 3600  # 1 hour timeout marker


def plot_runtime_scalability(ax, qtpu_df: pd.DataFrame, qac_df: pd.DataFrame, bench: str):
    """Plot 1: Bar plot comparing QTPU vs QAC total runtime."""
    sizes = sorted([s for s in qtpu_df["config.circuit_size"].unique() if s > 10])
    
    qtpu_bench = qtpu_df[qtpu_df["config.bench"] == bench]
    qac_bench = qac_df[qac_df["config.bench"] == bench]
    
    valid_sizes = []
    qtpu_times = []
    qac_times = []
    qac_timeout = []  # Track which QAC entries are timeouts
    
    for size in sizes:
        qtpu_row = qtpu_bench[qtpu_bench["config.circuit_size"] == size]
        qac_rows = qac_bench[qac_bench["config.circuit_size"] == size]
        
        if qtpu_row.empty or "result.quantum_time" not in qtpu_row.columns:
            continue
        
        qtpu_total = qtpu_row.iloc[0]["result.quantum_time"] + qtpu_row.iloc[0]["result.classical_time"]
        valid_sizes.append(size)
        qtpu_times.append(qtpu_total)
        
        # Check QAC availability - prefer non-timeout rows
        if not qac_rows.empty and "result.quantum_time" in qac_rows.columns:
            # Filter for rows that have valid quantum_time (non-timeout)
            valid_qac = qac_rows[qac_rows["result.quantum_time"].notna() & (qac_rows.get("result.timeout", False) != True)]
            if not valid_qac.empty:
                result = valid_qac.iloc[0]
                qac_total = result["result.quantum_time"] + result["result.classical_time"]
                qac_times.append(qac_total)
                qac_timeout.append(False)
            else:
                qac_times.append(TIMEOUT_VALUE)
                qac_timeout.append(True)
        else:
            # QAC doesn't have data for this size - assume timeout
            qac_times.append(TIMEOUT_VALUE)
            qac_timeout.append(True)
    
    if not valid_sizes:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(a) Runtime}")
        return
    
    x = np.arange(len(valid_sizes))
    width = 0.35
    
    # QTPU bars
    ax.bar(
        x - width / 2,
        qtpu_times,
        width,
        label=QTPU_LABEL,
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )
    
    # QAC bars - use different style for timeouts
    qac_colors = [colors()[1] if not to else "lightgray" for to in qac_timeout]
    bars = ax.bar(
        x + width / 2,
        qac_times,
        width,
        label=QAC_LABEL,
        color=qac_colors,
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )
    
    # Add timeout entry to legend if there are any timeouts
    if any(qac_timeout):
        from matplotlib.patches import Patch
        timeout_patch = Patch(facecolor="lightgray", edgecolor="black", hatch="\\\\", label="Timeout")
        ax.legend(handles=[ax.patches[0], bars[0], timeout_patch], labels=[QTPU_LABEL, QAC_LABEL, "Timeout"], loc="upper left")
    else:
        ax.legend(loc="upper left")
    
    ax.set_xlabel("Circuit Size ")
    ax.set_ylabel("Total Runtime [s]\n(lower is better)")
    ax.set_title(r"\textbf{(a) Runtime}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in valid_sizes])


def plot_memory_scalability(ax, qtpu_df: pd.DataFrame, qac_df: pd.DataFrame, bench: str):
    """Plot 2: Bar plot comparing number of circuits generated.
    
    QTPU: num_subcircuits
    QAC: num_subcircuits * 6^(circuit_size/10)
    """
    sizes = sorted([s for s in qtpu_df["config.circuit_size"].unique() if s > 10])
    
    qtpu_bench = qtpu_df[qtpu_df["config.bench"] == bench]
    
    valid_sizes = []
    qtpu_circuits = []
    qac_circuits = []
    
    for size in sizes:
        qtpu_row = qtpu_bench[qtpu_bench["config.circuit_size"] == size]
        
        if qtpu_row.empty or "result.num_subcircuits" not in qtpu_row.columns:
            continue
        
        # QTPU: just num_subcircuits
        num_subcirc = qtpu_row.iloc[0]["result.num_subcircuits"]
        valid_sizes.append(size)
        qtpu_circuits.append(num_subcirc)
        
        # QAC: num_subcircuits * 6^k where k = circuit_size / 10
        k = size // 10
        qac_circuits.append(num_subcirc * (6 ** k))
    
    if not valid_sizes:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(b) Generation Overhead}")
        return
    
    x = np.arange(len(valid_sizes))
    width = 0.35
    
    # QTPU bars
    ax.bar(
        x - width / 2,
        qtpu_circuits,
        width,
        label=QTPU_LABEL,
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )
    
    # QAC bars
    ax.bar(
        x + width / 2,
        qac_circuits,
        width,
        label=QAC_LABEL,
        color=colors()[1],
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )

    ax.set_xlabel("Circuit Size ")
    ax.set_ylabel("Circuits Generated\n(lower is better)")
    ax.set_title(r"\textbf{(b) Compilation Overhead}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in valid_sizes])


def plot_postprocessing_cost(ax, qtpu_df: pd.DataFrame, qac_df: pd.DataFrame, bench: str):
    """Plot 3: Bar plot comparing classical cost.
    
    QAC cost: 6^k where k = circuit_size / 10 (number of cuts)
    QTPU cost: num_experiments (tensor product size)
    """
    sizes = sorted([s for s in qtpu_df["config.circuit_size"].unique() if s > 10])
    
    qtpu_bench = qtpu_df[qtpu_df["config.bench"] == bench]
    
    valid_sizes = []
    qtpu_cost = []
    qac_cost = []
    
    for size in sizes:
        qtpu_row = qtpu_bench[qtpu_bench["config.circuit_size"] == size]
        
        if qtpu_row.empty or "result.num_experiments" not in qtpu_row.columns:
            continue
        
        # QTPU cost: num_experiments reflects the tensor product size
        qtpu_experiments = qtpu_row.iloc[0]["result.num_experiments"]
        valid_sizes.append(size)
        qtpu_cost.append(qtpu_experiments)
        
        # QAC cost: 6^k where k = circuit_size / 10 (number of cuts)
        k = size // 10
        qac_cost.append(6 ** k)
    
    if not valid_sizes:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(c) Postprocessing Overhead}")
        return
    
    x = np.arange(len(valid_sizes))
    width = 0.35
    
    # QTPU bars
    ax.bar(
        x - width / 2,
        qtpu_cost,
        width,
        label=QTPU_LABEL,
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )
    
    # QAC bars
    ax.bar(
        x + width / 2,
        qac_cost,
        width,
        label=QAC_LABEL,
        color=colors()[1],
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )

    ax.set_xlabel("Circuit Size ")
    ax.set_ylabel("Class. Cost [FLOPs]\n(lower is better)")
    ax.set_title(r"\textbf{(c) Classical Cost}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in valid_sizes])


@bk.pplot
def plot_scale_comparison(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame, bench: str = "qnn"):
    """Plot scalability comparison between QTPU and QAC.

    Args:
        qtpu_df: DataFrame with QTPU benchmark results.
        qac_df: DataFrame with QAC benchmark results.
        bench: Benchmark name to plot (default: "qnn").
    Returns:
        A BenchKit Plot object comparing the two approaches.
    """
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.3))

    plot_runtime_scalability(axes[0], qtpu_df, qac_df, bench)
    plot_memory_scalability(axes[1], qtpu_df, qac_df, bench)
    plot_postprocessing_cost(axes[2], qtpu_df, qac_df, bench)

    return fig


if __name__ == "__main__":
    # Disable TeX for local testing
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    register_style("qtpu", PlotStyle(color=colors()[0], hatch="//"))
    register_style("qac", PlotStyle(color=colors()[1], hatch="\\\\"))

    # Load data
    qtpu_data = bk.load_log("logs/scale/qtpu.jsonl")
    qac_data = bk.load_log("logs/scale/qac.jsonl")

    if isinstance(qtpu_data, pd.DataFrame):
        qtpu_df = qtpu_data
    else:
        qtpu_df = pd.json_normalize(qtpu_data)

    if isinstance(qac_data, pd.DataFrame):
        qac_df = qac_data
    else:
        qac_df = pd.json_normalize(qac_data) if qac_data else pd.DataFrame()

    if qtpu_df.empty:
        print("No QTPU data found")
        exit(1)

    print(f"QTPU: {len(qtpu_df)} rows, QAC: {len(qac_df)} rows")
    print(f"QTPU circuit sizes: {sorted(qtpu_df['config.circuit_size'].unique())}")
    print(f"QAC circuit sizes: {sorted(qac_df['config.circuit_size'].unique()) if not qac_df.empty else []}")

    # Plot for each benchmark
    for bench in ["qnn"]:
        if bench in qtpu_df["config.bench"].values:
            fig = plot_scale_comparison(qtpu_df, qac_df, bench=bench)
            plt.tight_layout()
            plt.savefig(f"plots/scale_{bench}.pdf", bbox_inches="tight")
            plt.show()
