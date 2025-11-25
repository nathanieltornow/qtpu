"""
Plot comparison between QTPU and QAC circuit cutting.

Shows:
1. Pareto frontier: c_cost vs max_error
2. Compile time comparison
3. Scaling with circuit size
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchkit as bk
from benchkit.plot.config import (
    PlotStyle,
    colors,
    register_style,
    single_column_width,
)

# Style configuration
FONTSIZE = 9
plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "axes.titlesize": FONTSIZE + 1,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE - 1,
})

QTPU_COLOR = "#2E86AB"
QAC_COLOR = "#E94F37"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load QTPU, QTPU Pareto, and QAC benchmark results."""
    qtpu_df = bk.load_log("logs/compile/qtpu.jsonl")
    qtpu_pareto_df = bk.load_log("logs/compile/qtpu_pareto.jsonl")
    qac_df = bk.load_log("logs/compile/qac.jsonl")
    return qtpu_df, qtpu_pareto_df, qac_df


def compute_max_error(qtensor_errors: list[float]) -> float:
    """Compute max error from list of subcircuit errors."""
    if not qtensor_errors:
        return 0.0
    return max(qtensor_errors)


@bk.pplot("pareto_frontier", custom_rc={"text.usetex": False})
def plot_pareto_frontier(qtpu_pareto_df: pd.DataFrame, qac_df: pd.DataFrame, circuit_size: int = None):
    """Plot Pareto frontier: c_cost (x) vs max_error (y)."""
    if circuit_size is None:
        circuit_size = qac_df["config.circuit_size"].max()
    
    qtpu_pareto = qtpu_pareto_df[qtpu_pareto_df["config.circuit_size"] == circuit_size]
    qac = qac_df[qac_df["config.circuit_size"] == circuit_size]
    
    fig, ax = plt.subplots(figsize=(single_column_width(), 2.8))
    
    # Extract QTPU Pareto frontier points
    if len(qtpu_pareto) > 0:
        frontier = qtpu_pareto.iloc[0]["result.pareto_frontier"]
        qtpu_c_costs = [p["c_cost"] for p in frontier]
        qtpu_max_errors = [p["max_error"] for p in frontier]
        
        ax.plot(qtpu_c_costs, qtpu_max_errors, 'o-',
                color=QTPU_COLOR, label='QTPU', markersize=6, linewidth=1.5, alpha=0.8)
    
    # Extract QAC points - max error from qtensor_errors
    qac_c_costs = qac["result.c_cost"].values
    qac_max_errors = qac["result.qtensor_errors"].apply(compute_max_error).values
    
    ax.plot(qac_c_costs, qac_max_errors, 's-',
            color=QAC_COLOR, label='QAC', markersize=6, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Classical Cost (c_cost)")
    ax.set_ylabel("Max Subcircuit Error")
    ax.set_title(f"Pareto Frontier ({circuit_size} qubits)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    return fig


@bk.pplot("pareto_all_sizes", custom_rc={"text.usetex": False})
def plot_pareto_all_sizes(qtpu_pareto_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot Pareto frontiers for all circuit sizes."""
    sizes = sorted(qac_df["config.circuit_size"].unique())
    
    fig, axes = plt.subplots(1, len(sizes), figsize=(single_column_width() * len(sizes), 2.8))
    if len(sizes) == 1:
        axes = [axes]
    
    for ax, circuit_size in zip(axes, sizes):
        qtpu_pareto = qtpu_pareto_df[qtpu_pareto_df["config.circuit_size"] == circuit_size]
        qac = qac_df[qac_df["config.circuit_size"] == circuit_size]
        
        # Extract QTPU Pareto frontier points
        if len(qtpu_pareto) > 0:
            frontier = qtpu_pareto.iloc[0]["result.pareto_frontier"]
            qtpu_c_costs = [p["c_cost"] for p in frontier]
            qtpu_max_errors = [p["max_error"] for p in frontier]
            
            ax.plot(qtpu_c_costs, qtpu_max_errors, 'o-',
                    color=QTPU_COLOR, label='QTPU', markersize=5, linewidth=1.5, alpha=0.8)
        
        # Extract QAC points
        qac_c_costs = qac["result.c_cost"].values
        qac_max_errors = qac["result.qtensor_errors"].apply(compute_max_error).values
        
        ax.plot(qac_c_costs, qac_max_errors, 's-',
                color=QAC_COLOR, label='QAC', markersize=5, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel("Classical Cost")
        ax.set_ylabel("Max Error")
        ax.set_title(f"{circuit_size}q")
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


@bk.pplot("compile_time", custom_rc={"text.usetex": False})
def plot_compile_time(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame, circuit_size: int = None):
    """Plot compile time comparison."""
    if circuit_size is None:
        circuit_size = qtpu_df["config.circuit_size"].max()
    
    qtpu = qtpu_df[qtpu_df["config.circuit_size"] == circuit_size].copy()
    qac = qac_df[qac_df["config.circuit_size"] == circuit_size].copy()
    
    fig, ax = plt.subplots(figsize=(single_column_width(), 2.5))
    
    # Create labels
    qtpu["label"] = qtpu["config.max_sampling_cost"].apply(lambda x: f"cost≤{int(x)}")
    qac["label"] = qac["config.fraction"].apply(lambda x: f"1/{int(x)}")
    
    # Sort by compile time
    qtpu = qtpu.sort_values("config.max_sampling_cost")
    qac = qac.sort_values("config.fraction")
    
    # Plot bars
    x_qtpu = np.arange(len(qtpu))
    x_qac = np.arange(len(qac)) + len(qtpu) + 0.5
    
    ax.bar(x_qtpu, qtpu["result.compile_time"], color=QTPU_COLOR, alpha=0.8, label='QTPU')
    ax.bar(x_qac, qac["result.compile_time"], color=QAC_COLOR, alpha=0.8, label='QAC')
    
    # Labels
    all_x = list(x_qtpu) + list(x_qac)
    all_labels = list(qtpu["label"]) + list(qac["label"])
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    
    ax.set_ylabel("Compile Time (s)")
    ax.set_title(f"Compilation Speed ({circuit_size} qubits)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Speedup annotation
    avg_qtpu = qtpu["result.compile_time"].mean()
    avg_qac = qac["result.compile_time"].mean()
    if avg_qtpu > 0 and avg_qac > avg_qtpu:
        ax.text(0.95, 0.95, f"QTPU {avg_qac/avg_qtpu:.0f}x faster",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, fontweight='bold', color=QTPU_COLOR)
    
    return fig


@bk.pplot("scaling", custom_rc={"text.usetex": False})
def plot_scaling(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Plot compile time scaling with circuit size."""
    fig, axes = plt.subplots(1, 2, figsize=(single_column_width() * 2, 2.5))
    
    # Left: Compile time scaling
    ax1 = axes[0]
    qtpu_grouped = qtpu_df.groupby("config.circuit_size")["result.compile_time"].mean()
    qac_grouped = qac_df.groupby("config.circuit_size")["result.compile_time"].mean()
    
    ax1.plot(qtpu_grouped.index, qtpu_grouped.values, 'o-',
             color=QTPU_COLOR, label='QTPU', markersize=8, linewidth=2)
    ax1.plot(qac_grouped.index, qac_grouped.values, 's-',
             color=QAC_COLOR, label='QAC', markersize=8, linewidth=2)
    
    ax1.set_xlabel("Circuit Size (qubits)")
    ax1.set_ylabel("Avg Compile Time (s)")
    ax1.set_title("Compile Time Scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Right: Best quantum cost achieved
    ax2 = axes[1]
    qtpu_df = qtpu_df.copy()
    qac_df = qac_df.copy()
    qtpu_df["max_width"] = qtpu_df["result.qtensor_widths"].apply(lambda x: max(x) if x else 0)
    qac_df["max_width"] = qac_df["result.qtensor_widths"].apply(lambda x: max(x) if x else 0)
    
    qtpu_best = qtpu_df.groupby("config.circuit_size")["max_width"].min()
    qac_best = qac_df.groupby("config.circuit_size")["max_width"].min()
    
    ax2.plot(qtpu_best.index, qtpu_best.values, 'o-',
             color=QTPU_COLOR, label='QTPU', markersize=8, linewidth=2)
    ax2.plot(qac_best.index, qac_best.values, 's-',
             color=QAC_COLOR, label='QAC', markersize=8, linewidth=2)
    
    ax2.set_xlabel("Circuit Size (qubits)")
    ax2.set_ylabel("Min Max Subcircuit Width")
    ax2.set_title("Best Quantum Cost Achieved")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


@bk.pplot("full_comparison", custom_rc={"text.usetex": False})
def plot_full_comparison(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Generate full 2x2 comparison plot."""
    circuit_size = qtpu_df["config.circuit_size"].max()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Top-left: Pareto frontier
    ax = axes[0, 0]
    qtpu = qtpu_df[qtpu_df["config.circuit_size"] == circuit_size]
    qac = qac_df[qac_df["config.circuit_size"] == circuit_size]
    
    qtpu_max_width = qtpu["result.qtensor_widths"].apply(lambda x: max(x) if x else 0)
    qac_max_width = qac["result.qtensor_widths"].apply(lambda x: max(x) if x else 0)
    
    ax.scatter(qtpu_max_width, qtpu["result.num_qtensors"],
               s=80, c=QTPU_COLOR, marker='o', label='QTPU', alpha=0.8)
    ax.scatter(qac_max_width, qac["result.num_qtensors"],
               s=80, c=QAC_COLOR, marker='s', label='QAC', alpha=0.8)
    ax.set_xlabel("Max Subcircuit Width")
    ax.set_ylabel("# Subcircuits")
    ax.set_title(f"Decomposition ({circuit_size}q)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: Compile time bars
    ax = axes[0, 1]
    qtpu = qtpu.copy()
    qac = qac.copy()
    qtpu["label"] = qtpu["config.max_sampling_cost"].apply(lambda x: f"≤{int(x)}")
    qac["label"] = qac["config.max_qubits"].apply(lambda x: f"≤{int(x)}q")
    qtpu = qtpu.sort_values("config.max_sampling_cost")
    qac = qac.sort_values("config.max_qubits")
    
    x_qtpu = np.arange(len(qtpu))
    x_qac = np.arange(len(qac)) + len(qtpu) + 0.5
    ax.bar(x_qtpu, qtpu["result.compile_time"], color=QTPU_COLOR, alpha=0.8, label='QTPU')
    ax.bar(x_qac, qac["result.compile_time"], color=QAC_COLOR, alpha=0.8, label='QAC')
    ax.set_xticks(list(x_qtpu) + list(x_qac))
    ax.set_xticklabels(list(qtpu["label"]) + list(qac["label"]), rotation=45, ha='right')
    ax.set_ylabel("Compile Time (s)")
    ax.set_title(f"Compilation Speed ({circuit_size}q)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bottom-left: Scaling
    ax = axes[1, 0]
    qtpu_grouped = qtpu_df.groupby("config.circuit_size")["result.compile_time"].mean()
    qac_grouped = qac_df.groupby("config.circuit_size")["result.compile_time"].mean()
    ax.plot(qtpu_grouped.index, qtpu_grouped.values, 'o-', color=QTPU_COLOR, label='QTPU', markersize=8)
    ax.plot(qac_grouped.index, qac_grouped.values, 's-', color=QAC_COLOR, label='QAC', markersize=8)
    ax.set_xlabel("Circuit Size (qubits)")
    ax.set_ylabel("Avg Compile Time (s)")
    ax.set_title("Compile Time Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Bottom-right: Best quantum cost
    ax = axes[1, 1]
    qtpu_df = qtpu_df.copy()
    qac_df = qac_df.copy()
    qtpu_df["max_width"] = qtpu_df["result.qtensor_widths"].apply(lambda x: max(x) if x else 0)
    qac_df["max_width"] = qac_df["result.qtensor_widths"].apply(lambda x: max(x) if x else 0)
    qtpu_best = qtpu_df.groupby("config.circuit_size")["max_width"].min()
    qac_best = qac_df.groupby("config.circuit_size")["max_width"].min()
    ax.plot(qtpu_best.index, qtpu_best.values, 'o-', color=QTPU_COLOR, label='QTPU', markersize=8)
    ax.plot(qac_best.index, qac_best.values, 's-', color=QAC_COLOR, label='QAC', markersize=8)
    ax.set_xlabel("Circuit Size (qubits)")
    ax.set_ylabel("Min Max Subcircuit Width")
    ax.set_title("Best Quantum Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("QTPU vs QAC Circuit Cutting", fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def print_summary(qtpu_df: pd.DataFrame, qac_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("QTPU vs QAC COMPARISON")
    print("="*60)
    
    avg_qtpu = qtpu_df["result.compile_time"].mean()
    avg_qac = qac_df["result.compile_time"].mean()
    
    print(f"\n⏱️  Compile Time:")
    print(f"   QTPU: {avg_qtpu:.2f}s avg")
    print(f"   QAC:  {avg_qac:.2f}s avg")
    print(f"   → QTPU is {avg_qac/avg_qtpu:.0f}x faster")
    
    qtpu_df = qtpu_df.copy()
    qac_df = qac_df.copy()
    qtpu_df["max_width"] = qtpu_df["result.qtensor_widths"].apply(lambda x: max(x) if x else 0)
    qac_df["max_width"] = qac_df["result.qtensor_widths"].apply(lambda x: max(x) if x else 0)
    
    print(f"\n📐 Max Subcircuit Width:")
    print(f"   QTPU: {qtpu_df['max_width'].min()}-{qtpu_df['max_width'].max()} qubits")
    print(f"   QAC:  {qac_df['max_width'].min()}-{qac_df['max_width'].max()} qubits")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    register_style("qtpu", PlotStyle(marker="o", color=QTPU_COLOR))
    register_style("qac", PlotStyle(marker="s", color=QAC_COLOR))
    
    qtpu_df, qtpu_pareto_df, qac_df = load_data()
    print_summary(qtpu_df, qac_df)
    
    plot_pareto_frontier(qtpu_pareto_df, qac_df)
    plot_pareto_all_sizes(qtpu_pareto_df, qac_df)
    plot_compile_time(qtpu_df, qac_df)
    plot_scaling(qtpu_df, qac_df)
    plot_full_comparison(qtpu_df, qac_df)
