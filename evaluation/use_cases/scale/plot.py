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


def plot_runtime_scalability(ax, qtpu_df: pd.DataFrame, qac_df: pd.DataFrame, bench: str):
    """Plot 1: Bar plot comparing QTPU vs QAC (inf samples) total runtime."""
    sizes = sorted(qtpu_df["config.circuit_size"].unique())
    
    qtpu_bench = qtpu_df[qtpu_df["config.bench"] == bench]
    qac_bench = qac_df[qac_df["config.bench"] == bench]
    
    # Filter QAC for inf samples only (stored as large float ~1.79e308)
    if "config.samples" in qac_bench.columns:
        qac_bench = qac_bench[qac_bench["config.samples"] > 1e300]
    
    # Find common sizes where both have data
    valid_sizes = []
    qtpu_times = []
    qac_times = []
    
    for size in sizes:
        qtpu_row = qtpu_bench[qtpu_bench["config.circuit_size"] == size]
        qac_row = qac_bench[qac_bench["config.circuit_size"] == size]
        
        if qtpu_row.empty or "result.quantum_time" not in qtpu_row.columns:
            continue
        
        qtpu_total = qtpu_row.iloc[0]["result.quantum_time"] + qtpu_row.iloc[0]["result.classical_time"]
        
        # Check QAC availability
        if not qac_row.empty and "result.quantum_time" in qac_row.columns:
            result = qac_row.iloc[0]
            # Check for timeout (failed column may not exist)
            if result.get("result.timeout", False):
                continue
            if pd.isna(result.get("result.quantum_time")):
                continue
            qac_total = result["result.quantum_time"] + result["result.classical_time"]
            valid_sizes.append(size)
            qtpu_times.append(qtpu_total)
            qac_times.append(qac_total)
    
    if not valid_sizes:
        ax.text(0.5, 0.5, "No overlapping data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(a) Runtime Comparison}")
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
    
    # QAC bars
    ax.bar(
        x + width / 2,
        qac_times,
        width,
        label=QAC_LABEL,
        color=colors()[1],
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )

    ax.set_xlabel("Circuit Size [qubits]")
    ax.set_ylabel("Total Runtime [s]")
    ax.set_title(r"\textbf{(a) Runtime Comparison}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in valid_sizes])


def plot_memory_scalability(ax, qtpu_df: pd.DataFrame, qac_df: pd.DataFrame, bench: str):
    """Plot 2: Bar plot comparing QTPU vs QAC (inf samples) memory usage."""
    sizes = sorted(qtpu_df["config.circuit_size"].unique())
    
    qtpu_bench = qtpu_df[qtpu_df["config.bench"] == bench]
    qac_bench = qac_df[qac_df["config.bench"] == bench]
    
    # Filter QAC for inf samples only (stored as large float ~1.79e308)
    if "config.samples" in qac_bench.columns:
        qac_bench = qac_bench[qac_bench["config.samples"] > 1e300]
    
    # Find common sizes where both have data
    valid_sizes = []
    qtpu_mem = []
    qac_mem = []
    
    for size in sizes:
        qtpu_row = qtpu_bench[qtpu_bench["config.circuit_size"] == size]
        qac_row = qac_bench[qac_bench["config.circuit_size"] == size]
        
        if qtpu_row.empty or "result.generation_memory" not in qtpu_row.columns:
            continue
        
        qtpu_mem_mb = qtpu_row.iloc[0]["result.generation_memory"] / (1024 * 1024)
        
        # Check QAC availability
        if not qac_row.empty and "result.generation_memory" in qac_row.columns:
            result = qac_row.iloc[0]
            if result.get("result.timeout", False):
                continue
            if pd.isna(result.get("result.generation_memory")):
                continue
            qac_mem_mb = result["result.generation_memory"] / (1024 * 1024)
            valid_sizes.append(size)
            qtpu_mem.append(qtpu_mem_mb)
            qac_mem.append(qac_mem_mb)
    
    if not valid_sizes:
        ax.text(0.5, 0.5, "No overlapping data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(b) Memory Usage}")
        return
    
    x = np.arange(len(valid_sizes))
    width = 0.35
    
    # QTPU bars
    ax.bar(
        x - width / 2,
        qtpu_mem,
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
        qac_mem,
        width,
        label=QAC_LABEL,
        color=colors()[1],
        edgecolor="black",
        linewidth=1,
        hatch="\\\\",
    )

    ax.set_xlabel("Circuit Size [qubits]")
    ax.set_ylabel("Peak Memory [MB]")
    ax.set_title(r"\textbf{(b) Memory Usage}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in valid_sizes])


def plot_runtime_breakdown(ax, qtpu_df: pd.DataFrame, qac_df: pd.DataFrame, bench: str):
    """Plot 3: Stacked bar chart showing runtime breakdown."""
    # Pick a few representative sizes
    all_sizes = sorted(qtpu_df["config.circuit_size"].unique())
    # Use subset of sizes for readability
    sizes = [s for s in all_sizes if s in [20, 40, 60, 80]]
    if not sizes:
        sizes = all_sizes[:4]

    x = np.arange(len(sizes))
    width = 0.35

    qtpu_quantum = []
    qtpu_classical = []
    qac_quantum = []
    qac_classical = []

    qtpu_bench = qtpu_df[qtpu_df["config.bench"] == bench]
    qac_bench = qac_df[qac_df["config.bench"] == bench]
    
    # Filter QAC for inf samples only (stored as large float ~1.79e308)
    if "config.samples" in qac_bench.columns:
        qac_bench = qac_bench[qac_bench["config.samples"] > 1e300]

    for size in sizes:
        # QTPU
        row = qtpu_bench[qtpu_bench["config.circuit_size"] == size]
        if not row.empty and "result.quantum_time" in row.columns:
            qtpu_quantum.append(row.iloc[0]["result.quantum_time"])
            qtpu_classical.append(row.iloc[0]["result.classical_time"])
        else:
            qtpu_quantum.append(0)
            qtpu_classical.append(0)

        # QAC
        row = qac_bench[qac_bench["config.circuit_size"] == size]
        if not row.empty and "result.quantum_time" in row.columns:
            result = row.iloc[0]
            if result.get("result.timeout", False) or result.get("result.failed", False):
                qac_quantum.append(0)
                qac_classical.append(0)
            elif pd.isna(result.get("result.quantum_time")):
                qac_quantum.append(0)
                qac_classical.append(0)
            else:
                qac_quantum.append(result["result.quantum_time"])
                qac_classical.append(result["result.classical_time"])
        else:
            qac_quantum.append(0)
            qac_classical.append(0)

    # QTPU bars
    ax.bar(
        x - width / 2,
        qtpu_quantum,
        width,
        label=f"{QTPU_LABEL} Quantum",
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
    )
    ax.bar(
        x - width / 2,
        qtpu_classical,
        width,
        bottom=qtpu_quantum,
        label=f"{QTPU_LABEL} Classical",
        color=colors()[0],
        edgecolor="black",
        linewidth=1,
        alpha=0.5,
        hatch="//",
    )

    # QAC bars
    ax.bar(
        x + width / 2,
        qac_quantum,
        width,
        label=f"{QAC_LABEL} Quantum",
        color=colors()[1],
        edgecolor="black",
        linewidth=1,
    )
    ax.bar(
        x + width / 2,
        qac_classical,
        width,
        bottom=qac_quantum,
        label=f"{QAC_LABEL} Classical",
        color=colors()[1],
        edgecolor="black",
        linewidth=1,
        alpha=0.5,
        hatch="\\\\",
    )

    ax.set_xlabel("Circuit Size [qubits]")
    ax.set_ylabel("Runtime [s]")
    ax.set_title(r"\textbf{(c) Runtime Breakdown}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in sizes])
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")


@bk.pplot
def plot_scale_comparison(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame, bench: str = "wstate"):
    """Plot scalability comparison between QTPU and QAC.

    Args:
        qtpu_df: DataFrame with QTPU benchmark results.
        qac_df: DataFrame with QAC benchmark results.
        bench: Benchmark name to plot (default: "wstate").
    Returns:
        A BenchKit Plot object comparing the two approaches.
    """
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.6))

    plot_runtime_scalability(axes[0], qtpu_df, qac_df, bench)
    plot_memory_scalability(axes[1], qtpu_df, qac_df, bench)
    plot_runtime_breakdown(axes[2], qtpu_df, qac_df, bench)

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
    print(f"QTPU columns: {list(qtpu_df.columns)}")

    # Plot for each benchmark
    for bench in ["wstate", "qnn"]:
        if bench in qtpu_df["config.bench"].values:
            fig = plot_scale_comparison(qtpu_df, qac_df, bench=bench)
            plt.tight_layout()
            plt.savefig(f"plots/scale_{bench}.pdf", bbox_inches="tight")
            plt.show()
