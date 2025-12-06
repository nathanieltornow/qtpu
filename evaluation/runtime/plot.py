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
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.4))
    
    bar_width = 0.2
    
    # ==========================================================================
    # (a) Runtime by Circuit Size (1 QPU) - bar chart with exponential reference
    # ==========================================================================
    # Use common circuit sizes for bar chart
    sizes_a = [40, 60, 80, 100]
    x_a = np.arange(len(sizes_a))
    bar_width_a = 0.2
    
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
        for size in sizes_a:
            row = bench_df[bench_df["config.circuit_size"] == size]
            if not row.empty:
                runtimes.append(compute_parallel_runtime(row.iloc[0], 1))
            else:
                runtimes.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [i for i, r in enumerate(runtimes) if not np.isnan(r)]
        valid_runtimes = [runtimes[i] for i in valid_indices]
        valid_x = x_a[valid_indices] + (benchmark_idx - 1.5) * bar_width_a
        
        axes[0].bar(
            valid_x,
            valid_runtimes,
            bar_width_a,
            label=BENCH_NAMES.get(bench, bench),
            color=colors()[benchmark_idx],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[benchmark_idx % len(HATCHES)],
        )
        benchmark_idx += 1
    
    # Dist-VQE (cluster_size=15) - only has 100q and 150q
    dist_df = dist_qtpu_df[dist_qtpu_df["config.cluster_size"] == 15].copy()
    if not dist_df.empty:
        runtimes = []
        for size in sizes_a:
            row = dist_df[dist_df["config.circuit_size"] == size]
            if not row.empty:
                runtimes.append(compute_parallel_runtime(row.iloc[0], 1))
            else:
                runtimes.append(np.nan)
        
        valid_indices = [i for i, r in enumerate(runtimes) if not np.isnan(r)]
        valid_runtimes = [runtimes[i] for i in valid_indices]
        valid_x = x_a[valid_indices] + (benchmark_idx - 1.5) * bar_width_a
        
        axes[0].bar(
            valid_x,
            valid_runtimes,
            bar_width_a,
            label=BENCH_NAMES.get("dist-vqe", "Dist-VQE"),
            color=colors()[benchmark_idx],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[benchmark_idx % len(HATCHES)],
        )
    
    axes[0].set_xlabel("Circuit Size (qubits)")
    axes[0].set_ylabel("Runtime [s]")
    axes[0].set_xticks(x_a)
    axes[0].set_xticklabels([f"{s}q" for s in sizes_a])
    axes[0].set_title(r"\textbf{(a) Runtime Scaling}", fontsize=9)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # ==========================================================================
    # (b) Speedup by Number of QPUs - bar chart showing near-linear scaling
    # ==========================================================================
    qpu_counts = [1, 2, 4, 8, 16]  # Only up to 16 QPUs
    x_b = np.arange(len(qpu_counts))
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
        & (dist_qtpu_df["config.cluster_size"] == 15)
    ]
    if not dist_row.empty:
        benchmark_data.append(("dist-vqe", dist_row.iloc[0]))
    
    for i, (bench, row) in enumerate(benchmark_data):
        runtime_1 = compute_parallel_runtime(row, 1)
        speedups = [runtime_1 / compute_parallel_runtime(row, n) for n in qpu_counts]
        
        offset = (i - len(benchmark_data)/2 + 0.5) * bar_width
        axes[1].bar(
            x_b + offset,
            speedups,
            bar_width,
            label=BENCH_NAMES.get(bench, bench),
            color=colors()[i],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[i % len(HATCHES)],
        )
    
    axes[1].set_xlabel("Number of QPUs")
    axes[1].set_ylabel(r"Speedup [$\times$]")
    axes[1].set_xticks(x_b)
    axes[1].set_xticklabels([str(n) for n in qpu_counts])
    axes[1].set_title(r"\textbf{(b) Speedup}", fontsize=9)
    axes[1].set_ylim(0, max(qpu_counts) * 1.1)
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # Add ideal linear scaling line (after other plots so it appears last in legend)
    axes[1].plot(x_b, qpu_counts, "k--", linewidth=1.5, label="Linear", zorder=10)
    
    # Create a single shared legend below both (a) and (b)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.35, -0.05), 
              fontsize=6, ncol=5, frameon=True)
    
    # ==========================================================================
    # (c) Runtime by Cluster Size (Dist-VQE, 100q) - QTPU vs cuTensorNet
    # ==========================================================================
    cluster_sizes = sorted(dist_qtpu_df["config.cluster_size"].unique())
    cluster_sizes = [cs for cs in cluster_sizes if cs < 20 and cs != 15]  # Limit to 20q clusters, exclude 15
    x_c = np.arange(len(cluster_sizes))
    num_qpus = 4
    bar_width_c = 0.35
    
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
    axes[2].bar(
        x_c - bar_width_c/2,
        qtpu_runtimes,
        bar_width_c,
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
        axes[2].bar(
            x_c + bar_width_c/2,
            success_runtimes,
            bar_width_c,
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
                max_y = axes[2].get_ylim()[1] if axes[2].get_ylim()[1] > 0 else max(qtpu_runtimes) * 1.2
                axes[2].bar(
                    x_c[i] + bar_width_c/2,
                    max_y * 0.9,
                    bar_width_c,
                    color='lightgray',
                    edgecolor="black",
                    linewidth=0.5,
                    hatch='xxx',
                )
                axes[2].annotate('OOM', xy=(x_c[i] + bar_width_c/2, max_y * 0.45),
                               ha='center', va='center', fontsize=5, fontweight='bold',
                               color='red', rotation=90)
    
    axes[2].set_xlabel("Cluster Size")
    axes[2].set_ylabel("Runtime [s]")
    axes[2].set_xticks(x_c)
    axes[2].set_xticklabels([f"{int(cs)}q" for cs in cluster_sizes])
    axes[2].set_title(r"\textbf{(c) " + QTPU_LABEL + r" vs Classical}", fontsize=9)
    axes[2].legend(loc="upper left", fontsize=6)
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].set_yscale("log")
    
    # ==========================================================================
    # Print summary statistics
    # ==========================================================================
    print("\n" + "="*80)
    print("RUNTIME ANALYSIS SUMMARY")
    print("="*80)
    
    # (a) Runtime scaling summary
    print("\n(a) Runtime Scaling (1 QPU, 100q):")
    print("-" * 80)
    for bench in STANDARD_BENCHMARKS:
        row = qtpu_df[
            (qtpu_df["config.bench"] == bench)
            & (qtpu_df["config.circuit_size"] == 100)
            & (qtpu_df["config.cluster_size"] == DEFAULT_CLUSTER_SIZE)
        ]
        if not row.empty:
            runtime = compute_parallel_runtime(row.iloc[0], 1)
            print(f"  {BENCH_NAMES.get(bench, bench):<10} {runtime:>8.2f}s")
    
    # Dist-VQE
    dist_row = dist_qtpu_df[
        (dist_qtpu_df["config.circuit_size"] == 100)
        & (dist_qtpu_df["config.cluster_size"] == 15)
    ]
    if not dist_row.empty:
        runtime = compute_parallel_runtime(dist_row.iloc[0], 1)
        print(f"  {BENCH_NAMES.get('dist-vqe', 'Dist-VQE'):<10} {runtime:>8.2f}s")
    
    # (b) Speedup analysis
    print("\n(b) Speedup Analysis (100q, W-State as example):")
    print("-" * 80)
    wstate_row = qtpu_df[
        (qtpu_df["config.bench"] == "wstate")
        & (qtpu_df["config.circuit_size"] == 100)
        & (qtpu_df["config.cluster_size"] == DEFAULT_CLUSTER_SIZE)
    ]
    if not wstate_row.empty:
        r = wstate_row.iloc[0]
        runtime_1 = compute_parallel_runtime(r, 1)
        qpu_counts_test = [1, 2, 4, 8, 16]
        print(f"  {'QPUs':<8} {'Runtime [s]':<12} {'Speedup':<10} {'Efficiency':<12}")
        for n in qpu_counts_test:
            runtime_n = compute_parallel_runtime(r, n)
            speedup = runtime_1 / runtime_n
            efficiency = speedup / n * 100
            print(f"  {n:<8} {runtime_n:<12.2f} {speedup:<10.2f}x {efficiency:<12.1f}%")
    
    # Runtime breakdown
    print("\n(c) Runtime Breakdown (100q, cluster_size=15):")
    print("-" * 80)
    print(f"{'Benchmark':<12} {'Compile':<10} {'Quantum':<10} {'Classical':<12} {'Quantum %':<12}")
    print("-" * 80)
    
    for bench in STANDARD_BENCHMARKS:
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
            qpct = qt / total * 100
            print(f"{BENCH_NAMES.get(bench, bench):<12} {ct:<10.3f} {qt:<10.2f} {cct:<12.4f} {qpct:<12.1f}")
    
    # Classical vs QTPU comparison
    if dist_classical_df is not None and not dist_classical_df.empty:
        print("\n(d) QTPU vs Classical (Dist-VQE, 100q, 4 QPUs):")
        print("-" * 80)
        print(f"{'Cluster':<10} {'QTPU [s]':<12} {'Classical [s]':<15} {'Speedup':<12} {'Status':<10}")
        print("-" * 80)
        
        for cs in sorted(dist_qtpu_df["config.cluster_size"].unique()):
            if cs >= 20 or cs == 15:
                continue
                
            qtpu_row = dist_qtpu_df[
                (dist_qtpu_df["config.circuit_size"] == 100)
                & (dist_qtpu_df["config.cluster_size"] == cs)
            ]
            
            if not qtpu_row.empty:
                qtpu_time = compute_parallel_runtime(qtpu_row.iloc[0], 4)
                
                classical_row = dist_classical_df[
                    (dist_classical_df["config.circuit_size"] == 100)
                    & (dist_classical_df["config.cluster_size"] == cs)
                    & (dist_classical_df["result.status"].notna())
                ]
                
                if not classical_row.empty:
                    r = classical_row.iloc[0]
                    if r["result.status"] == "success":
                        classical_time = r["result.contract_time"]
                        speedup = classical_time / qtpu_time
                        print(f"{int(cs)}q{'':<6} {qtpu_time:<12.2f} {classical_time:<15.2f} {speedup:<12.2f}x {'Success':<10}")
                    else:
                        print(f"{int(cs)}q{'':<6} {qtpu_time:<12.2f} {'OOM':<15} {'-':<12} {'OOM':<10}")
                else:
                    print(f"{int(cs)}q{'':<6} {qtpu_time:<12.2f} {'N/A':<15} {'-':<12} {'No data':<10}")
    
    print("\n" + "="*80 + "\n")
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=0.5)
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
