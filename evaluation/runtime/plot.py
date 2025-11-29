"""Plotting for runtime scalability evaluation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchkit as bk
from benchkit.plot.config import (
    double_column_width,
    single_column_width,
    colors,
    register_style,
    PlotStyle,
)


BENCH_NAMES = {
    "qnn": "QNN",
    "wstate": "W-State",
    "vqe_su2": "VQE-SU2",
}

QPU_COLORS = {
    1: colors()[0],
    2: colors()[1],
    4: colors()[2],
    8: colors()[3],
    16: colors()[4],
    32: colors()[5] if len(colors()) > 5 else colors()[0],
}


def plot_e2e_runtime_by_qpus(
    ax,
    df: pd.DataFrame,
    bench: str,
    title: str = None,
    show_legend: bool = True,
):
    """Plot end-to-end runtime vs circuit size for different QPU counts."""
    bench_df = df[df["config.bench"] == bench]
    
    qpu_counts = sorted(bench_df["config.num_qpus"].unique())
    
    for num_qpus in qpu_counts:
        qpu_df = bench_df[bench_df["config.num_qpus"] == num_qpus]
        qpu_df = qpu_df.sort_values("config.circuit_size")
        
        sizes = qpu_df["config.circuit_size"].values
        times = qpu_df["result.total_time"].values
        
        color = QPU_COLORS.get(num_qpus, colors()[0])
        ax.plot(
            sizes,
            times,
            "o-",
            color=color,
            markersize=6,
            linewidth=1.5,
            label=f"{num_qpus} QPUs",
        )
    
    ax.set_xlabel("Circuit Size (qubits)")
    ax.set_ylabel("End-to-End Runtime [s]")
    ax.set_yscale("log")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_runtime_breakdown(
    ax,
    df: pd.DataFrame,
    bench: str,
    num_qpus: int = 4,
    title: str = None,
):
    """Stacked bar chart showing runtime breakdown (compile, quantum, classical)."""
    bench_df = df[
        (df["config.bench"] == bench) & 
        (df["config.num_qpus"] == num_qpus)
    ].sort_values("config.circuit_size")
    
    sizes = bench_df["config.circuit_size"].values
    compile_times = bench_df["result.compile_time"].values
    quantum_times = bench_df["result.quantum_time_parallel"].values
    classical_times = bench_df["result.classical_contraction_time"].values
    
    x = np.arange(len(sizes))
    width = 0.6
    
    ax.bar(x, compile_times, width, label="Compile", color=colors()[0])
    ax.bar(x, quantum_times, width, bottom=compile_times, label="Quantum", color=colors()[1])
    ax.bar(x, classical_times, width, bottom=compile_times + quantum_times, label="Classical", color=colors()[2])
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in sizes])
    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Runtime [s]")
    ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_speedup_vs_qpus(
    ax,
    df: pd.DataFrame,
    bench: str,
    circuit_size: int = 100,
    title: str = None,
):
    """Plot speedup as number of QPUs increases for a fixed circuit size."""
    bench_df = df[
        (df["config.bench"] == bench) & 
        (df["config.circuit_size"] == circuit_size)
    ].sort_values("config.num_qpus")
    
    if bench_df.empty:
        return
    
    qpus = bench_df["config.num_qpus"].values
    times = bench_df["result.total_time"].values
    
    # Speedup relative to 1 QPU
    baseline = times[0] if len(times) > 0 else 1
    speedups = baseline / times
    
    # Ideal speedup (only quantum portion parallelizes)
    quantum_frac = bench_df["result.quantum_time_sequential"].values[0] / baseline if baseline > 0 else 0
    ideal_speedups = 1 / (1 - quantum_frac + quantum_frac / qpus)
    
    ax.plot(qpus, speedups, "o-", color=colors()[0], markersize=8, linewidth=2, label="Actual")
    ax.plot(qpus, ideal_speedups, "--", color=colors()[1], linewidth=2, label="Amdahl's Law")
    ax.plot(qpus, qpus, ":", color="gray", linewidth=1, label="Linear")
    
    ax.set_xlabel("Number of QPUs")
    ax.set_ylabel("Speedup")
    ax.set_xscale("log", base=2)
    ax.set_xticks(qpus)
    ax.set_xticklabels([str(q) for q in qpus])
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_compile_scalability(
    ax,
    df: pd.DataFrame,
    title: str = None,
    show_legend: bool = True,
):
    """Plot compilation time vs circuit size for all benchmarks."""
    benchmarks = df["config.bench"].unique()
    
    for i, bench in enumerate(benchmarks):
        bench_df = df[df["config.bench"] == bench].sort_values("config.circuit_size")
        
        sizes = bench_df["config.circuit_size"].values
        times = bench_df["result.compile_time"].values
        
        ax.plot(
            sizes,
            times,
            "o-",
            color=colors()[i],
            markersize=6,
            linewidth=1.5,
            label=BENCH_NAMES.get(bench, bench),
        )
    
    ax.set_xlabel("Circuit Size (qubits)")
    ax.set_ylabel("Compile Time [s]")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


@bk.pplot
def plot_runtime_analysis(df: pd.DataFrame):
    """Main runtime analysis figure.
    
    Creates a 2x3 grid showing:
    - Row 1: E2E runtime vs circuit size for each benchmark (varying QPUs)
    - Row 2: Runtime breakdown, speedup curve, compile scalability
    """
    benchmarks = ["qnn", "wstate", "vqe_su2"]
    fig, axes = plt.subplots(2, 3, figsize=(double_column_width(), 3.2))
    
    # Row 1: E2E runtime for each benchmark
    for i, bench in enumerate(benchmarks):
        plot_e2e_runtime_by_qpus(
            axes[0, i],
            df,
            bench,
            title=rf"\textbf{{({chr(ord('a') + i)}) {BENCH_NAMES[bench]}}}",
            show_legend=(i == 0),
        )
    
    # Row 2: Breakdown, speedup, compile time
    plot_runtime_breakdown(
        axes[1, 0],
        df,
        "vqe_su2",
        num_qpus=4,
        title=r"\textbf{(d) Runtime Breakdown (4 QPUs)}",
    )
    
    plot_speedup_vs_qpus(
        axes[1, 1],
        df,
        "vqe_su2",
        circuit_size=100,
        title=r"\textbf{(e) Speedup vs QPUs (100q)}",
    )
    
    # For compile scalability, use the compile_only data if available
    plot_compile_scalability(
        axes[1, 2],
        df,
        title=r"\textbf{(f) Compile Scalability}",
    )
    
    plt.tight_layout()
    return fig


@bk.pplot
def plot_qpu_scaling(df: pd.DataFrame):
    """Single-column plot showing QPU scaling for VQE-SU2."""
    fig, ax = plt.subplots(1, 1, figsize=(single_column_width(), 2.0))
    
    plot_e2e_runtime_by_qpus(
        ax,
        df,
        "vqe_su2",
        title=r"\textbf{VQE-SU2 Runtime Scaling}",
        show_legend=True,
    )
    
    plt.tight_layout()
    return fig


def plot_qtpu_vs_quimb(
    ax,
    qtpu_df: pd.DataFrame,
    quimb_df: pd.DataFrame,
    bench: str,
    num_qpus: int = 8,
    title: str = None,
    show_legend: bool = True,
):
    """Compare QTPU (with N QPUs) vs classical quimb execution."""
    # QTPU data for specific QPU count
    qtpu_bench = qtpu_df[
        (qtpu_df["config.bench"] == bench) &
        (qtpu_df["config.num_qpus"] == num_qpus)
    ].sort_values("config.circuit_size")
    
    # Quimb baseline
    quimb_bench = quimb_df[
        quimb_df["config.bench"] == bench
    ].sort_values("config.circuit_size")
    
    if not qtpu_bench.empty:
        ax.plot(
            qtpu_bench["config.circuit_size"].values,
            qtpu_bench["result.total_time"].values,
            "o-",
            color=colors()[0],
            markersize=6,
            linewidth=1.5,
            label=f"QTPU ({num_qpus} QPUs)",
        )
    
    if not quimb_bench.empty:
        ax.plot(
            quimb_bench["config.circuit_size"].values,
            quimb_bench["result.sample_time"].values,
            "s--",
            color=colors()[2],
            markersize=6,
            linewidth=1.5,
            label="quimb (classical)",
        )
    
    ax.set_xlabel("Circuit Size (qubits)")
    ax.set_ylabel("Runtime [s]")
    ax.set_yscale("log")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


@bk.pplot
def plot_comparison_with_quimb(qtpu_df: pd.DataFrame, quimb_df: pd.DataFrame):
    """Compare QTPU vs quimb across all benchmarks.
    
    Shows that QTPU with multiple QPUs can outperform classical tensor
    network contraction for larger circuit sizes.
    """
    benchmarks = ["qnn", "wstate", "vqe_su2"]
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 2.0))
    
    for i, bench in enumerate(benchmarks):
        plot_qtpu_vs_quimb(
            axes[i],
            qtpu_df,
            quimb_df,
            bench,
            num_qpus=8,
            title=rf"\textbf{{({chr(ord('a') + i)}) {BENCH_NAMES[bench]}}}",
            show_legend=(i == 0),
        )
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib
    
    # Disable TeX for testing
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"
    
    # Load data
    try:
        e2e_data = bk.load_log("logs/runtime/e2e_runtime.jsonl")
        df = pd.DataFrame(e2e_data)
        
        fig = plot_runtime_analysis(df)
        plt.show()
    except FileNotFoundError:
        print("No runtime data found. Run evaluation/runtime/run.py first.")
        print("  python evaluation/runtime/run.py e2e")
    
    # Try to load quimb baseline and plot comparison
    try:
        quimb_data = bk.load_log("logs/runtime/quimb_baseline.jsonl")
        quimb_df = pd.DataFrame(quimb_data)
        
        fig = plot_comparison_with_quimb(df, quimb_df)
        plt.show()
    except FileNotFoundError:
        print("No quimb baseline data found. Run:")
        print("  python evaluation/runtime/run.py quimb")
