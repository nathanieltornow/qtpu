#!/usr/bin/env python3
"""Generate all new plots for the OSDI revision.

This script generates:
1. Combined workload figure (knitting + PEC + batching)
2. Scale correctness figure (add to Fig 12)
3. Error mitigation effectiveness figure (add to Fig 14)
4. Hardware simulation results figure

Run after benchmarks complete:
    PYTHONPATH=. uv run python evaluation/plot_all_new.py
"""

import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import benchkit as bk
from benchkit.plot.config import colors, double_column_width

# Create output directory
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
OUTPUT_DIR = Path(f"plots/revision_{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_log_safe(path: str) -> pd.DataFrame:
    """Load a log file, returning empty DataFrame if not found."""
    try:
        data = bk.load_log(path)
        if isinstance(data, pd.DataFrame):
            return data
        return pd.json_normalize(data) if data else pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return pd.DataFrame()


# =============================================================================
# Plot 1: Combined Workload
# =============================================================================

def plot_combined_workload():
    """Generate combined workload comparison figure.

    Shows the composability advantage: qTPU composes cutting + mitigation + batching
    in a single hEinsum expression, while the baseline needs 3 frameworks.
    Panels: (a) circuit count, (b) E2E accuracy (from correctness data), (c) E2E time breakdown.
    """
    print("Generating combined workload plot...")

    qtpu_df = load_log_safe("logs/combined/qtpu.jsonl")
    baseline_df = load_log_safe("logs/combined/baseline.jsonl")
    correctness_df = load_log_safe("logs/scale/correctness.jsonl")

    if qtpu_df.empty:
        print("  No qTPU combined data found, skipping")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.5))

    # Get unique circuit sizes from combined benchmark
    sizes = sorted(qtpu_df["config.circuit_size"].unique())
    x = np.arange(len(sizes))
    width = 0.35

    # Helper: get first row for each size
    def get_row(df, size):
        rows = df[df["config.circuit_size"] == size]
        return rows.iloc[0] if not rows.empty else None

    # Panel (a): Circuit count comparison (log scale)
    qtpu_counts = []
    baseline_counts = []
    for size in sizes:
        qr = get_row(qtpu_df, size)
        qtpu_counts.append(qr.get("result.num_circuits", 1) if qr is not None else 1)
        if not baseline_df.empty:
            br = baseline_df[baseline_df["config.circuit_size"] == size]
            baseline_counts.append(br["result.num_circuits"].median() if not br.empty else 1)
        else:
            baseline_counts.append(1)

    axes[0].bar(x - width/2, qtpu_counts, width, label=r"\textsc{qTPU}", color=colors()[0], edgecolor="black", hatch="//")
    axes[0].bar(x + width/2, baseline_counts, width, label="QAC+Mitiq", color=colors()[1], edgecolor="black", hatch="\\\\")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Circuit Size")
    axes[0].set_ylabel("Circuits")
    axes[0].set_title(r"\textbf{(a) Circuit Count}")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}q" for s in sizes])
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel (b): E2E accuracy from correctness validation
    # This shows the end-to-end pipeline produces correct results
    if not correctness_df.empty and "result.absolute_error" in correctness_df.columns:
        # Use wstate results (meaningful absolute errors)
        wstate_df = correctness_df[correctness_df["config.bench"] == "wstate"] if "config.bench" in correctness_df.columns else correctness_df
        wstate_df = wstate_df[wstate_df["result.absolute_error"].notna()]

        corr_sizes = sorted(wstate_df["config.circuit_size"].unique())
        cx = np.arange(len(corr_sizes))
        means = []
        stds = []
        for cs in corr_sizes:
            errs = wstate_df[wstate_df["config.circuit_size"] == cs]["result.absolute_error"]
            means.append(errs.mean())
            stds.append(errs.std() if len(errs) > 1 else 0)

        axes[1].bar(cx, means, yerr=stds, capsize=3, color=colors()[0], edgecolor="black")
        axes[1].set_xlabel("Circuit Size")
        axes[1].set_ylabel("Absolute Error")
        axes[1].set_title(r"\textbf{(b) E2E Accuracy}")
        axes[1].set_xticks(cx)
        axes[1].set_xticklabels([f"{s}q" for s in corr_sizes])
        for i, (mean, std) in enumerate(zip(means, stds)):
            axes[1].annotate(f"{mean:.1e}", xy=(i, mean + std), xytext=(0, 3),
                           textcoords="offset points", ha="center", fontsize=6)
    else:
        axes[1].text(0.5, 0.5, "No correctness data", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title(r"\textbf{(b) E2E Accuracy}")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Panel (c): E2E time breakdown for qTPU
    qtpu_compile = []
    qtpu_quantum = []
    qtpu_classical = []
    for size in sizes:
        qr = get_row(qtpu_df, size)
        qtpu_compile.append(qr.get("result.compile_time", 0) if qr is not None else 0)
        qtpu_quantum.append(qr.get("result.quantum_time", 0) if qr is not None else 0)
        qtpu_classical.append(qr.get("result.classical_time", 0) if qr is not None else 0)

    bar_width = 0.6
    axes[2].bar(x, qtpu_compile, bar_width, label="Compile", color=colors()[0], edgecolor="black")
    axes[2].bar(x, qtpu_quantum, bar_width, bottom=qtpu_compile, label="QPU (est.)", color=colors()[1], edgecolor="black")
    axes[2].bar(x, qtpu_classical, bar_width,
                bottom=[c+q for c,q in zip(qtpu_compile, qtpu_quantum)], label="Classical", color=colors()[2], edgecolor="black")
    axes[2].set_xlabel("Circuit Size")
    axes[2].set_ylabel("Time [s]")
    axes[2].set_title(r"\textbf{(c) E2E Time}")
    axes[2].legend(loc="upper left", fontsize=6)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"{s}q" for s in sizes])
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "combined_workload.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "combined_workload.png", bbox_inches="tight", dpi=150)
    print(f"  Saved to {OUTPUT_DIR / 'combined_workload.pdf'}")
    return fig


# =============================================================================
# Plot 2: Scale Correctness
# =============================================================================

def plot_scale_correctness():
    """Generate scale correctness validation figure.

    Shows absolute error of cut circuit reconstruction vs ideal simulation.
    Groups by benchmark circuit type (wstate, qnn) and circuit size.
    """
    print("Generating scale correctness plot...")

    df = load_log_safe("logs/scale/correctness.jsonl")

    if df.empty:
        print("  No correctness data found, skipping")
        return None

    # Filter valid rows
    if "result.skipped" in df.columns:
        df = df[df["result.skipped"] != True].copy()
    if "result.absolute_error" in df.columns:
        df = df[df["result.absolute_error"].notna()].copy()

    benchmarks = sorted(df["config.bench"].unique()) if "config.bench" in df.columns else ["qnn"]
    sizes = sorted(df["config.circuit_size"].unique())

    fig, ax = plt.subplots(figsize=(3.5, 2))

    x = np.arange(len(sizes))
    n_bench = len(benchmarks)
    width = 0.7 / max(n_bench, 1)

    for i, bench in enumerate(benchmarks):
        bench_df = df[df["config.bench"] == bench] if "config.bench" in df.columns else df
        means = []
        stds = []
        for size in sizes:
            subset = bench_df[bench_df["config.circuit_size"] == size]
            errors = subset["result.absolute_error"].dropna() if not subset.empty else pd.Series(dtype=float)
            means.append(errors.mean() if len(errors) > 0 else np.nan)
            stds.append(errors.std() if len(errors) > 1 else 0)

        offset = (i - (n_bench - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=bench, color=colors()[i], edgecolor="black")

        for j, (bar, mean) in enumerate(zip(bars, means)):
            if not np.isnan(mean):
                ax.annotate(f"{mean:.1e}",
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6)

    ax.set_xlabel("Circuit Size [qubits]")
    ax.set_ylabel("Absolute Error\n(lower is better)")
    ax.set_title(r"\textbf{Correctness Validation}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in sizes])
    if n_bench > 1:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "scale_correctness.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "scale_correctness.png", bbox_inches="tight", dpi=150)
    print(f"  Saved to {OUTPUT_DIR / 'scale_correctness.pdf'}")
    return fig


# =============================================================================
# Plot 3: Error Mitigation Effectiveness
# =============================================================================

def plot_mitigation_effectiveness():
    """Generate error mitigation effectiveness figure.

    Shows ZNE (Zero Noise Extrapolation) improves accuracy on noisy circuits.
    Panels: (a) error by circuit size, (b) ZNE improvement factor, (c) error vs noise.
    """
    print("Generating mitigation effectiveness plot...")

    df = load_log_safe("logs/error_mitigation/effectiveness.jsonl")

    if df.empty:
        print("  No effectiveness data found, skipping")
        return None

    # Filter out error/skipped rows
    if "result.absolute_error" in df.columns:
        df = df[df["result.absolute_error"].notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.5))

    circuit_sizes = sorted(df["config.circuit_size"].unique())
    noise_levels = sorted(df["config.noise_level"].unique())

    # Panel (a): Error comparison across circuit sizes (at middle noise level)
    noise_level = noise_levels[len(noise_levels)//2] if noise_levels else 0.01
    subset = df[df["config.noise_level"] == noise_level]

    x = np.arange(len(circuit_sizes))
    width = 0.35
    none_errs = []
    zne_errs = []
    for cs in circuit_sizes:
        none_row = subset[(subset["config.circuit_size"] == cs) & (subset["config.mitigation"] == "none")]
        zne_row = subset[(subset["config.circuit_size"] == cs) & (subset["config.mitigation"] == "zne")]
        none_errs.append(none_row.iloc[0]["result.absolute_error"] if not none_row.empty else np.nan)
        zne_errs.append(zne_row.iloc[0]["result.absolute_error"] if not zne_row.empty else np.nan)

    axes[0].bar(x - width/2, none_errs, width, label="Unmitigated", color=colors()[1], edgecolor="black")
    axes[0].bar(x + width/2, zne_errs, width, label="ZNE", color=colors()[0], edgecolor="black")
    axes[0].set_xlabel("Circuit Size")
    axes[0].set_ylabel("Absolute Error\n(lower is better)")
    axes[0].set_title(rf"\textbf{{(a) Error (p={noise_level})}}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}q" for s in circuit_sizes])
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel (b): ZNE improvement factor across all configs
    improvements = []
    for cs in circuit_sizes:
        for nl in noise_levels:
            sub = df[(df["config.circuit_size"] == cs) & (df["config.noise_level"] == nl)]
            none_row = sub[sub["config.mitigation"] == "none"]
            zne_row = sub[sub["config.mitigation"] == "zne"]
            if not none_row.empty and not zne_row.empty:
                baseline_err = none_row.iloc[0].get("result.absolute_error", 0)
                zne_err = zne_row.iloc[0].get("result.absolute_error", 0)
                if zne_err > 0 and baseline_err > 0:
                    improvements.append(baseline_err / zne_err)

    if improvements:
        bp = axes[1].boxplot([improvements], labels=["ZNE"], patch_artist=True,
                             boxprops=dict(facecolor=colors()[0], alpha=0.7))
        axes[1].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        median_val = np.median(improvements)
        axes[1].annotate(f"median: {median_val:.1f}$\\times$",
                        xy=(1, median_val), xytext=(1.3, median_val),
                        fontsize=7, fontweight="bold")
    axes[1].set_ylabel(r"Improvement ($\times$)")
    axes[1].set_title(r"\textbf{(b) ZNE Improvement}")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Panel (c): Error vs noise level for largest circuit
    circuit_size = circuit_sizes[-1] if circuit_sizes else 8
    for method, label, color, marker in [("none", "Unmitigated", colors()[1], "s"), ("zne", "ZNE", colors()[0], "o")]:
        errs = []
        for nl in noise_levels:
            row = df[(df["config.circuit_size"] == circuit_size) & (df["config.noise_level"] == nl) & (df["config.mitigation"] == method)]
            if not row.empty:
                errs.append(row.iloc[0].get("result.absolute_error", np.nan))
            else:
                errs.append(np.nan)
        axes[2].plot(noise_levels, errs, f"{marker}-", color=color, label=label, markersize=5)

    axes[2].set_xlabel("Noise Level")
    axes[2].set_ylabel("Absolute Error")
    axes[2].set_title(rf"\textbf{{(c) Error vs Noise ({circuit_size}q)}}")
    axes[2].legend(loc="upper left", fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "mitigation_effectiveness.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "mitigation_effectiveness.png", bbox_inches="tight", dpi=150)
    print(f"  Saved to {OUTPUT_DIR / 'mitigation_effectiveness.pdf'}")
    return fig


# =============================================================================
# Plot 4: Hardware Simulation Results
# =============================================================================

def plot_hardware_results():
    """Generate hardware simulation results figure.

    Shows qTPU circuit cutting on noisy hardware (FakeBrisbane noise model).
    Panels: (a) absolute error qTPU vs direct, (b) error vs subcircuit size, (c) QPU time.
    """
    print("Generating hardware results plot...")

    qtpu_df = load_log_safe("logs/hardware/qtpu_noisy.jsonl")
    direct_df = load_log_safe("logs/hardware/direct_noisy.jsonl")

    if qtpu_df.empty:
        print("  No hardware data found, skipping")
        return None

    # Filter out skipped/error rows
    if "result.skipped" in qtpu_df.columns:
        qtpu_df = qtpu_df[qtpu_df["result.skipped"] != True].copy()

    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.5))

    sizes = sorted(qtpu_df["config.circuit_size"].unique())

    # Panel (a): Absolute error comparison (best subcircuit size for qTPU)
    qtpu_errors = []
    direct_errors = []
    for size in sizes:
        qtpu_row = qtpu_df[qtpu_df["config.circuit_size"] == size]
        if not qtpu_row.empty and "result.absolute_error" in qtpu_row.columns:
            valid = qtpu_row[qtpu_row["result.absolute_error"].notna()]
            if not valid.empty:
                best = valid.loc[valid["result.absolute_error"].idxmin()]
                qtpu_errors.append(best["result.absolute_error"])
            else:
                qtpu_errors.append(np.nan)
        else:
            qtpu_errors.append(np.nan)

        direct_row = direct_df[direct_df["config.circuit_size"] == size] if not direct_df.empty else pd.DataFrame()
        if not direct_row.empty and "result.absolute_error" in direct_row.columns:
            direct_errors.append(direct_row.iloc[0]["result.absolute_error"])
        else:
            direct_errors.append(np.nan)

    x = np.arange(len(sizes))
    width = 0.35
    axes[0].bar(x - width/2, qtpu_errors, width, label=r"\textsc{qTPU}", color=colors()[0], edgecolor="black", hatch="//")
    axes[0].bar(x + width/2, direct_errors, width, label="Direct", color=colors()[1], edgecolor="black", hatch="\\\\")
    axes[0].set_xlabel("Circuit Size")
    axes[0].set_ylabel("Absolute Error\n(lower is better)")
    axes[0].set_title(r"\textbf{(a) Noisy Simulation}")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}q" for s in sizes])
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel (b): Error vs subcircuit size (shows cutting helps on noisy hardware)
    circuit_size = sizes[-1] if sizes else 20
    sub_df = qtpu_df[(qtpu_df["config.circuit_size"] == circuit_size) & (qtpu_df["result.absolute_error"].notna())]
    if not sub_df.empty:
        subcirc_sizes = sorted(sub_df["config.subcirc_size"].unique())
        abs_errs = []
        for sc in subcirc_sizes:
            row = sub_df[sub_df["config.subcirc_size"] == sc]
            if not row.empty:
                abs_errs.append(row.iloc[0]["result.absolute_error"])
            else:
                abs_errs.append(np.nan)
        axes[1].plot(subcirc_sizes, abs_errs, "o-", color=colors()[0], markersize=6, label=r"\textsc{qTPU}")
        # Add direct baseline as horizontal line
        if not direct_df.empty:
            direct_row = direct_df[direct_df["config.circuit_size"] == circuit_size]
            if not direct_row.empty:
                direct_err = direct_row.iloc[0]["result.absolute_error"]
                axes[1].axhline(y=direct_err, color=colors()[1], linestyle="--", label="Direct", alpha=0.8)
        axes[1].set_xlabel("Subcircuit Size [qubits]")
        axes[1].set_ylabel("Absolute Error")
        axes[1].set_title(rf"\textbf{{(b) Cut Size ({circuit_size}q)}}")
        axes[1].legend(loc="best", fontsize=7)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title(r"\textbf{(b) Cut Size}")

    # Panel (c): QPU time estimate
    estimated = []
    simulated = []
    for size in sizes:
        row = qtpu_df[qtpu_df["config.circuit_size"] == size]
        if not row.empty:
            est = row.iloc[0].get("result.estimated_qpu_time", np.nan)
            sim = row.iloc[0].get("result.sim_time", np.nan)
            if not np.isnan(est):
                estimated.append(est)
                simulated.append(sim)
            else:
                estimated.append(0)
                simulated.append(0)

    if estimated and simulated:
        axes[2].bar(x - width/2, estimated, width, label="Estimated QPU", color=colors()[0], edgecolor="black")
        axes[2].bar(x + width/2, simulated, width, label="Sim. Wall Time", color=colors()[1], edgecolor="black")
        axes[2].set_xlabel("Circuit Size")
        axes[2].set_ylabel("Time [s]")
        axes[2].set_title(r"\textbf{(c) QPU Time}")
        axes[2].legend(loc="upper left", fontsize=7)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"{s}q" for s in sizes])
        axes[2].grid(True, alpha=0.3, axis="y")
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title(r"\textbf{(c) QPU Time}")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "hardware_results.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "hardware_results.png", bbox_inches="tight", dpi=150)
    print(f"  Saved to {OUTPUT_DIR / 'hardware_results.pdf'}")
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Generating revision plots to {OUTPUT_DIR}")
    print("=" * 60)

    # Configure matplotlib for publication quality
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.size"] = 9
    matplotlib.rcParams["axes.titlesize"] = 10
    matplotlib.rcParams["axes.labelsize"] = 9
    matplotlib.rcParams["xtick.labelsize"] = 8
    matplotlib.rcParams["ytick.labelsize"] = 8
    matplotlib.rcParams["legend.fontsize"] = 7

    # Generate all plots
    plot_combined_workload()
    plot_scale_correctness()
    plot_mitigation_effectiveness()
    plot_hardware_results()

    print("=" * 60)
    print(f"Done! All plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
