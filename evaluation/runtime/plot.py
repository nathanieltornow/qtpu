"""Plotting for runtime scalability evaluation.

This module creates plots for QTPU runtime analysis showing superscalability:
1. Speedup bar chart showing near-linear scaling with QPU count
2. Runtime breakdown showing parallelizable vs serial components
3. Throughput scaling demonstrating superlinear benefits

Key insight: The quantum workload is embarrassingly parallel across subcircuits.
With N QPUs, quantum_time_parallel = quantum_time_sequential / N.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchkit as bk
from benchkit.plot.config import (
    double_column_width,
    single_column_width,
    colors,
)


QTPU_LABEL = r"\textsc{qTPU}"

# Benchmark display names
BENCH_NAMES = {
    "qnn": "QNN",
    "wstate": "W-State",
    "vqe_su2": "VQE",
    "dist-vqe": "Dist-VQE",
}

# All 4 benchmarks
ALL_BENCHMARKS = ["qnn", "wstate", "vqe_su2", "dist-vqe"]
STANDARD_BENCHMARKS = ["qnn", "wstate", "vqe_su2"]

# Fixed cluster size for scalability analysis
DEFAULT_CLUSTER_SIZE = 15

# QPU counts for scalability analysis
QPU_COUNTS = [1, 2, 4, 8, 16, 32]

# Circuit sizes
CIRCUIT_SIZES = [20, 50, 100, 150]

# Hatches for different benchmarks (consistent across all plots)
HATCHES = ['', '///', '...', 'xxx']


# =============================================================================
# Data Loading Utilities
# =============================================================================


def load_standard_qtpu() -> pd.DataFrame:
    """Load and preprocess standard QTPU benchmark data."""
    df = bk.load_log("logs/runtime/standard_qtpu.jsonl")
    # Filter out failed runs (where result is NaN or compile_time is missing)
    df = df[df["result.compile_time"].notna()]
    return df


def load_standard_classical() -> pd.DataFrame:
    """Load and preprocess standard classical benchmark data."""
    df = bk.load_log("logs/runtime/standard_classical.jsonl")
    return df


def load_dist_vqe_qtpu() -> pd.DataFrame:
    """Load and preprocess distributed VQE QTPU benchmark data."""
    df = bk.load_log("logs/runtime/dist_vqe_qtpu.jsonl")
    # Filter out failed runs
    df = df[df["result.compile_time"].notna()]
    return df


def load_dist_vqe_classical() -> pd.DataFrame:
    """Load and preprocess distributed VQE classical benchmark data."""
    df = bk.load_log("logs/runtime/dist_vqe_classical.jsonl")
    return df


def compute_parallel_runtime(row: pd.Series, num_qpus: int) -> float:
    """Compute total runtime with N parallel QPUs.
    
    The quantum workload is embarrassingly parallel:
    - quantum_time is the sequential time (all subcircuits run one after another)
    - With N QPUs, quantum portion takes quantum_time / N
    - Compile and classical contraction are serial overheads
    """
    compile_time = row["result.compile_time"]
    quantum_time_seq = row["result.quantum_time"]
    classical_time = row["result.classical_contraction_time"]
    
    quantum_time_parallel = quantum_time_seq / num_qpus
    return compile_time + quantum_time_parallel + classical_time


# =============================================================================
# Plot Functions (matching compiler/plot.py style)
# =============================================================================


def plot_e2e_runtime_by_benchmark(
    ax,
    qtpu_df: pd.DataFrame,
    benchmarks: list = None,
    sizes: list = None,
    num_qpus: int = 4,
    cluster_size: int = DEFAULT_CLUSTER_SIZE,
    title: str = None,
    show_legend: bool = True,
):
    """Plot end-to-end runtime vs circuit size for multiple benchmarks.
    
    Args:
        ax: Matplotlib axis.
        qtpu_df: DataFrame with QTPU results.
        benchmarks: List of benchmark names to plot.
        sizes: List of circuit sizes to include.
        num_qpus: Number of QPUs for parallel execution.
        cluster_size: Cluster size to filter by.
        title: Plot title.
        show_legend: Whether to show legend.
    """
    if benchmarks is None:
        benchmarks = STANDARD_BENCHMARKS
    if sizes is None:
        sizes = CIRCUIT_SIZES

    for i, bench in enumerate(benchmarks):
        bench_df = qtpu_df[
            (qtpu_df["config.bench"] == bench)
            & (qtpu_df["config.cluster_size"] == cluster_size)
        ].copy()

        if bench_df.empty:
            continue

        # Compute parallel runtime
        bench_df["runtime"] = bench_df.apply(
            lambda row: compute_parallel_runtime(row, num_qpus), axis=1
        )
        bench_df = bench_df.sort_values("config.circuit_size")

        # Filter to requested sizes
        bench_df = bench_df[bench_df["config.circuit_size"].isin(sizes)]

        ax.plot(
            bench_df["config.circuit_size"].values,
            bench_df["runtime"].values,
            "o-",
            color=colors()[i],
            markersize=8,
            linewidth=2,
            markeredgecolor="black",
            markeredgewidth=1,
            label=BENCH_NAMES.get(bench, bench),
        )

    ax.set_xlabel("Circuit Size (qubits)")
    ax.set_ylabel("Runtime [s]")
    ax.set_yscale("log")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}q" for s in sizes])


def plot_runtime_by_qpu_count(
    ax,
    qtpu_df: pd.DataFrame,
    benchmarks: list = None,
    circuit_size: int = 100,
    qpu_counts: list = None,
    cluster_size: int = DEFAULT_CLUSTER_SIZE,
    title: str = None,
    show_legend: bool = True,
):
    """Plot runtime vs number of QPUs for 100q benchmarks.
    
    Args:
        ax: Matplotlib axis.
        qtpu_df: DataFrame with QTPU results.
        benchmarks: List of benchmark names to plot.
        circuit_size: Circuit size to use (default 100).
        qpu_counts: List of QPU counts to evaluate.
        cluster_size: Cluster size to filter by.
        title: Plot title.
        show_legend: Whether to show legend.
    """
    if benchmarks is None:
        benchmarks = STANDARD_BENCHMARKS
    if qpu_counts is None:
        qpu_counts = QPU_COUNTS

    for i, bench in enumerate(benchmarks):
        row = qtpu_df[
            (qtpu_df["config.bench"] == bench)
            & (qtpu_df["config.circuit_size"] == circuit_size)
            & (qtpu_df["config.cluster_size"] == cluster_size)
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        # Compute runtime for each QPU count
        runtimes = [compute_parallel_runtime(row, n) for n in qpu_counts]

        ax.plot(
            qpu_counts,
            runtimes,
            "o-",
            color=colors()[i],
            markersize=8,
            linewidth=2,
            markeredgecolor="black",
            markeredgewidth=1,
            label=BENCH_NAMES.get(bench, bench),
        )

    ax.set_xlabel("Number of QPUs")
    ax.set_ylabel("Runtime [s]")
    ax.set_yscale("log")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(qpu_counts)
    ax.set_xticklabels([str(n) for n in qpu_counts])


def plot_dist_vqe_by_cluster_size(
    ax,
    dist_qtpu_df: pd.DataFrame,
    cluster_sizes: list = None,
    num_qpus: int = 4,
    title: str = None,
    show_legend: bool = True,
):
    """Plot dist-VQE runtime for different cluster sizes with fixed QPU count.
    
    Args:
        ax: Matplotlib axis.
        dist_qtpu_df: DataFrame with dist-VQE QTPU results.
        cluster_sizes: List of cluster sizes to plot.
        num_qpus: Number of QPUs for parallel execution.
        title: Plot title.
        show_legend: Whether to show legend.
    """
    if cluster_sizes is None:
        cluster_sizes = [10, 15, 20]

    for i, cluster_size in enumerate(cluster_sizes):
        cluster_df = dist_qtpu_df[
            dist_qtpu_df["config.cluster_size"] == cluster_size
        ].copy()

        if cluster_df.empty:
            continue

        # Compute parallel runtime
        cluster_df["runtime"] = cluster_df.apply(
            lambda row: compute_parallel_runtime(row, num_qpus), axis=1
        )
        cluster_df = cluster_df.sort_values("config.circuit_size")

        ax.plot(
            cluster_df["config.circuit_size"].values,
            cluster_df["runtime"].values,
            "o-",
            color=colors()[i],
            markersize=8,
            linewidth=2,
            markeredgecolor="black",
            markeredgewidth=1,
            label=f"cluster={cluster_size}",
        )

    ax.set_xlabel("Circuit Size (qubits)")
    ax.set_ylabel("Runtime [s]")
    ax.set_yscale("log")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


# =============================================================================
# Main Plot Function
# =============================================================================


@bk.pplot
def plot_runtime_analysis(qtpu_df: pd.DataFrame, dist_qtpu_df: pd.DataFrame, dist_classical_df: pd.DataFrame = None):
    """Main runtime analysis figure (1 row, 4 columns) with bar charts.
    
    Creates a 1x4 grid showing superscalability:
    - (a) Runtime by circuit size (1 QPU baseline)
    - (b) Runtime breakdown (percentage) sharing y-axis with (a)
    - (c) Speedup by number of QPUs (showing near-linear scaling)
    - (d) Runtime by cluster size (dist-VQE) comparing QTPU vs cuTensorNet

    Args:
        qtpu_df: DataFrame with standard QTPU results.
        dist_qtpu_df: DataFrame with dist-VQE QTPU results.
        dist_classical_df: DataFrame with dist-VQE classical (cuTensorNet) results.
    Returns:
        A BenchKit Plot object.
    """
    # Create figure with 4 subplots, second one thinner for breakdown
    fig, axes = plt.subplots(1, 4, figsize=(double_column_width(), 1.8),
                              gridspec_kw={'width_ratios': [1, 0.5, 1, 1]})
    
    bar_width = 0.2
    
    # ==========================================================================
    # (a) Runtime by Circuit Size (1 QPU) - bar chart
    # ==========================================================================
    sizes = [20, 50, 100]
    x = np.arange(len(sizes))
    
    benchmark_idx = 0
    # Standard benchmarks
    for bench in STANDARD_BENCHMARKS:
        bench_df = qtpu_df[
            (qtpu_df["config.bench"] == bench)
            & (qtpu_df["config.cluster_size"] == DEFAULT_CLUSTER_SIZE)
        ].copy()
        if bench_df.empty:
            continue
        
        runtimes = []
        for size in sizes:
            row = bench_df[bench_df["config.circuit_size"] == size]
            if not row.empty:
                runtimes.append(compute_parallel_runtime(row.iloc[0], 1))
            else:
                runtimes.append(0)
        
        offset = (benchmark_idx - 1.5) * bar_width
        axes[0].bar(
            x + offset,
            runtimes,
            bar_width,
            label=BENCH_NAMES.get(bench, bench),
            color=colors()[benchmark_idx],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[benchmark_idx % len(HATCHES)],
        )
        benchmark_idx += 1
    
    # Dist-VQE (cluster_size=10)
    dist_df = dist_qtpu_df[dist_qtpu_df["config.cluster_size"] == 10].copy()
    if not dist_df.empty:
        runtimes = []
        for size in sizes:
            row = dist_df[dist_df["config.circuit_size"] == size]
            if not row.empty:
                runtimes.append(compute_parallel_runtime(row.iloc[0], 1))
            else:
                runtimes.append(0)
        
        offset = (benchmark_idx - 1.5) * bar_width
        axes[0].bar(
            x + offset,
            runtimes,
            bar_width,
            label=BENCH_NAMES.get("dist-vqe", "Dist-VQE"),
            color=colors()[benchmark_idx],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[benchmark_idx % len(HATCHES)],
        )
    
    axes[0].set_xlabel("Circuit Size")
    axes[0].set_ylabel("Runtime [s]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}q" for s in sizes])
    axes[0].set_title(r"\textbf{(a) Runtime}" + "\n" + r"\textit{lower is better $\downarrow$}", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=5)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # ==========================================================================
    # (b) Runtime Breakdown (percentage stacked bar) - shares y concept with (a)
    # ==========================================================================
    breakdown_benchmarks = ["qnn", "wstate"]
    x_b = np.arange(len(breakdown_benchmarks))
    bar_width_b = 0.6
    
    # For percentage breakdown, normalize to 100%
    compile_pcts = []
    quantum_pcts = []
    classical_pcts = []
    
    for bench in breakdown_benchmarks:
        row = qtpu_df[
            (qtpu_df["config.bench"] == bench)
            & (qtpu_df["config.circuit_size"] == 100)
            & (qtpu_df["config.cluster_size"] == DEFAULT_CLUSTER_SIZE)
        ]
        if not row.empty:
            r = row.iloc[0]
            ct = r["result.compile_time"]
            qt = r["result.quantum_time"]
            cct = r["result.classical_contraction_time"]
            total = ct + qt + cct
            compile_pcts.append(ct / total * 100)
            quantum_pcts.append(qt / total * 100)
            classical_pcts.append(cct / total * 100)
        else:
            compile_pcts.append(0)
            quantum_pcts.append(0)
            classical_pcts.append(0)
    
    # Stacked percentage bar chart with hatches (Quantum at bottom, Compile, Classical on top)
    axes[1].bar(x_b, quantum_pcts, bar_width_b, label="Quantum", 
                color=colors()[1], edgecolor="black", linewidth=0.5, hatch='')
    axes[1].bar(x_b, compile_pcts, bar_width_b, bottom=quantum_pcts, 
                label="Compile", color=colors()[0], edgecolor="black", linewidth=0.5, hatch='///')
    bottom = [q + c for q, c in zip(quantum_pcts, compile_pcts)]
    axes[1].bar(x_b, classical_pcts, bar_width_b, bottom=bottom, 
                label="Classical", color=colors()[2], edgecolor="black", linewidth=0.5, hatch='...')
    
    # Add percentage labels - inside if they fit, above otherwise
    for i, (qp, cp, clp) in enumerate(zip(quantum_pcts, compile_pcts, classical_pcts)):
        # Quantum label - always fits inside (it's >90%)
        axes[1].annotate(f'{qp:.1f}%', xy=(i, qp/2), ha='center', va='center', 
                        fontsize=6, color='white', fontweight='bold')
        
        # Compile label - inside if > 5%, otherwise above bar
        if cp > 5:
            axes[1].annotate(f'{cp:.1f}%', xy=(i, qp + cp/2), ha='center', va='center', 
                            fontsize=6, color='white', fontweight='bold')
        else:
            axes[1].annotate(f'{cp:.1f}%', xy=(i, 103), ha='center', va='bottom', 
                            fontsize=5, color=colors()[0])
    
    # Single classical label in the middle top
    mid_x = (x_b[0] + x_b[-1]) / 2
    axes[1].annotate(f'Classical: <0.01%', xy=(mid_x, 110), ha='center', va='bottom', 
                    fontsize=5, color=colors()[2])
    
    axes[1].set_ylabel("Fraction [\\%]")
    axes[1].set_xticks(x_b)
    axes[1].set_xticklabels([BENCH_NAMES.get(b, b) for b in breakdown_benchmarks], fontsize=6)
    axes[1].set_title(r"\textbf{(b) Breakdown}" + "\n" + r"\textit{~}", fontsize=8)
    axes[1].legend(loc="lower right", fontsize=5)
    axes[1].set_ylim(0, 120)  # Extra space for labels above
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # ==========================================================================
    # (c) Speedup by Number of QPUs - bar chart showing near-linear scaling
    # ==========================================================================
    qpu_counts = [1, 2, 4, 8, 16]  # Only up to 16 QPUs
    x_c = np.arange(len(qpu_counts))
    circuit_size = 100
    
    # Collect data for all benchmarks at 100q
    benchmark_data = []
    for bench in STANDARD_BENCHMARKS:
        row = qtpu_df[
            (qtpu_df["config.bench"] == bench)
            & (qtpu_df["config.circuit_size"] == circuit_size)
            & (qtpu_df["config.cluster_size"] == DEFAULT_CLUSTER_SIZE)
        ]
        if not row.empty:
            benchmark_data.append((bench, row.iloc[0]))
    
    # Dist-VQE at 100q
    dist_row = dist_qtpu_df[
        (dist_qtpu_df["config.circuit_size"] == 100)
        & (dist_qtpu_df["config.cluster_size"] == 10)
    ]
    if not dist_row.empty:
        benchmark_data.append(("dist-vqe", dist_row.iloc[0]))
    
    for i, (bench, row) in enumerate(benchmark_data):
        runtime_1 = compute_parallel_runtime(row, 1)
        speedups = [runtime_1 / compute_parallel_runtime(row, n) for n in qpu_counts]
        
        offset = (i - len(benchmark_data)/2 + 0.5) * bar_width
        axes[2].bar(
            x_c + offset,
            speedups,
            bar_width,
            label=BENCH_NAMES.get(bench, bench),
            color=colors()[i],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[i % len(HATCHES)],
        )
    
    # Add ideal linear scaling line
    axes[2].plot(x_c, qpu_counts, "k--", linewidth=1.5, label="Linear", zorder=10)
    
    axes[2].set_xlabel("Number of QPUs")
    axes[2].set_ylabel(r"Speedup [$\times$]")
    axes[2].set_xticks(x_c)
    axes[2].set_xticklabels([str(n) for n in qpu_counts])
    axes[2].set_title(r"\textbf{(c) Speedup}" + "\n" + r"\textit{higher is better $\uparrow$}", fontsize=8)
    axes[2].legend(loc="upper left", fontsize=5, ncol=1)
    axes[2].set_ylim(0, max(qpu_counts) * 1.1)
    axes[2].grid(True, alpha=0.3, axis="y")
    
    # ==========================================================================
    # (d) Runtime by Cluster Size (Dist-VQE, 100q) - QTPU vs cuTensorNet
    # ==========================================================================
    cluster_sizes = sorted(dist_qtpu_df["config.cluster_size"].unique())
    cluster_sizes = [cs for cs in cluster_sizes if cs < 20]  # Limit to 20q clusters
    x_d = np.arange(len(cluster_sizes))
    num_qpus = 4
    bar_width_d = 0.35
    
    # Get QTPU runtime for each cluster size at 100q (with 4 QPUs)
    qtpu_runtimes = []
    for cs in cluster_sizes:
        row = dist_qtpu_df[
            (dist_qtpu_df["config.circuit_size"] == 100)
            & (dist_qtpu_df["config.cluster_size"] == cs)
        ]
        if not row.empty:
            qtpu_runtimes.append(compute_parallel_runtime(row.iloc[0], num_qpus))
        else:
            qtpu_runtimes.append(0)
    
    # Get classical (cuTensorNet) runtime for each cluster size at 100q
    classical_runtimes = []
    classical_oom = []  # Track which ones failed with OOM
    if dist_classical_df is not None:
        for cs in cluster_sizes:
            row = dist_classical_df[
                (dist_classical_df["config.circuit_size"] == 100)
                & (dist_classical_df["config.cluster_size"] == cs)
                & (dist_classical_df["result.status"].notna())
            ]
            if not row.empty:
                r = row.iloc[0]
                if r["result.status"] == "success":
                    classical_runtimes.append(r["result.contract_time"])
                    classical_oom.append(False)
                else:
                    # Failed (OOM) - use a placeholder value for bar height
                    classical_runtimes.append(max(qtpu_runtimes) * 1.5 if qtpu_runtimes else 100)
                    classical_oom.append(True)
            else:
                classical_runtimes.append(0)
                classical_oom.append(False)
    
    # Plot QTPU bars
    axes[3].bar(
        x_d - bar_width_d/2,
        qtpu_runtimes,
        bar_width_d,
        label=QTPU_LABEL,
        color=colors()[0],
        edgecolor="black",
        linewidth=0.5,
        hatch=HATCHES[0],
    )
    
    # Plot classical bars (if available)
    if dist_classical_df is not None and classical_runtimes:
        # Plot successful bars
        success_runtimes = [r if not oom else 0 for r, oom in zip(classical_runtimes, classical_oom)]
        axes[3].bar(
            x_d + bar_width_d/2,
            success_runtimes,
            bar_width_d,
            label="cuTensorNet",
            color=colors()[1],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[1],
        )
        
        # Add OOM markers for failed bars
        for i, (cs, oom) in enumerate(zip(cluster_sizes, classical_oom)):
            if oom:
                # Draw a hatched bar at max height to indicate OOM
                max_y = axes[3].get_ylim()[1] if axes[3].get_ylim()[1] > 0 else max(qtpu_runtimes) * 1.2
                axes[3].bar(
                    x_d[i] + bar_width_d/2,
                    max_y * 0.9,
                    bar_width_d,
                    color='lightgray',
                    edgecolor="black",
                    linewidth=0.5,
                    hatch='xxx',
                )
                axes[3].annotate('OOM', xy=(x_d[i] + bar_width_d/2, max_y * 0.45),
                               ha='center', va='center', fontsize=5, fontweight='bold',
                               color='red', rotation=90)
    
    axes[3].set_xlabel("Cluster Size")
    axes[3].set_ylabel("Runtime [s]")
    axes[3].set_xticks(x_d)
    axes[3].set_xticklabels([f"{int(cs)}q" for cs in cluster_sizes])
    axes[3].set_title(r"\textbf{(d) " + QTPU_LABEL + r" vs Classical}" + "\n" + r"\textit{lower is better $\downarrow$}", fontsize=8)
    axes[3].legend(loc="upper left", fontsize=5)
    axes[3].grid(True, alpha=0.3, axis="y")
    axes[3].set_yscale("log")
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import matplotlib
    import os

    # Must disable TeX FIRST before any plot functions called
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    import matplotlib.pyplot as plt

    print("Loading data...")

    # Load standard benchmark data
    qtpu_df = None
    dist_qtpu_df = None

    if os.path.exists("logs/runtime/standard_qtpu.jsonl"):
        qtpu_df = load_standard_qtpu()
        print(f"Loaded {len(qtpu_df)} standard QTPU entries")
        print(f"  Benchmarks: {qtpu_df['config.bench'].unique().tolist()}")
        print(f"  Cluster sizes: {sorted(qtpu_df['config.cluster_size'].unique())}")
        print(f"  Circuit sizes: {sorted(qtpu_df['config.circuit_size'].unique())}")

    if os.path.exists("logs/runtime/dist_vqe_qtpu.jsonl"):
        dist_qtpu_df = load_dist_vqe_qtpu()
        print(f"Loaded {len(dist_qtpu_df)} dist-VQE QTPU entries")
        print(f"  Cluster sizes: {sorted(dist_qtpu_df['config.cluster_size'].unique())}")

    dist_classical_df = None
    if os.path.exists("logs/runtime/dist_vqe_classical.jsonl"):
        dist_classical_df = load_dist_vqe_classical()
        # Filter to get latest run per config (keeping most recent)
        dist_classical_df = dist_classical_df.sort_values('timestamp').drop_duplicates(
            subset=['config.circuit_size', 'config.cluster_size'], keep='last'
        )
        print(f"Loaded {len(dist_classical_df)} dist-VQE classical entries")

    # Generate main figure
    if qtpu_df is not None and dist_qtpu_df is not None:
        fig = plot_runtime_analysis(qtpu_df, dist_qtpu_df, dist_classical_df)
        plt.show()
    else:
        print("Missing data. Run:")
        print("  python evaluation/runtime/run.py standard")
        print("  python evaluation/runtime/run.py dist")
