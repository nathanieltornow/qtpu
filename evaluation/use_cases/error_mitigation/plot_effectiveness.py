"""Plotting for Error Mitigation Effectiveness Benchmark.

Creates a 3-panel figure showing:
(a) Error Reduction: Unmitigated vs Mitigated vs Ideal
(b) Mitigation Improvement Factor by Method
(c) Error vs Noise Level
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchkit.plot.config import (
    colors,
    double_column_width,
)

import benchkit as bk


def plot_error_comparison(ax, df: pd.DataFrame):
    """Panel (a): Compare unmitigated vs mitigated expectation values."""
    # Get data for median circuit size and noise level
    circuit_sizes = sorted(df["config.circuit_size"].unique())
    noise_levels = sorted(df["config.noise_level"].unique())

    circuit_size = circuit_sizes[len(circuit_sizes) // 2] if circuit_sizes else 6
    noise_level = noise_levels[len(noise_levels) // 2] if noise_levels else 0.01

    subset = df[
        (df["config.circuit_size"] == circuit_size) &
        (df["config.noise_level"] == noise_level)
    ]

    if subset.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"\textbf{(a) Error Comparison}")
        return

    methods = ["none", "zne", "twirl"]
    method_labels = ["Unmitigated", "ZNE", "Twirl"]

    ideal = subset[subset["config.mitigation"] == "none"].iloc[0].get("result.ideal_expval", 0) if len(subset[subset["config.mitigation"] == "none"]) > 0 else 0

    errors = []
    for method in methods:
        row = subset[subset["config.mitigation"] == method]
        if not row.empty and "result.absolute_error" in row.columns:
            errors.append(row.iloc[0].get("result.absolute_error", 1.0))
        else:
            errors.append(1.0)

    x = np.arange(len(methods))
    bars = ax.bar(x, errors, color=[colors()[1], colors()[0], colors()[2]], edgecolor="black")

    ax.set_xlabel("Method")
    ax.set_ylabel("Absolute Error\n(lower is better)")
    ax.set_title(rf"\textbf{{(a) Error ({circuit_size}q, noise={noise_level})}}")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate improvement
    if len(errors) >= 2 and errors[0] > 0:
        for i, (e, label) in enumerate(zip(errors[1:], method_labels[1:])):
            improvement = errors[0] / max(e, 1e-10)
            ax.annotate(
                f"{improvement:.1f}×",
                xy=(i + 1, e),
                xytext=(i + 1, e + 0.05),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )


def plot_improvement_factor(ax, df: pd.DataFrame):
    """Panel (b): Mitigation improvement factor by method."""
    methods = ["zne", "twirl"]
    method_labels = ["ZNE", "Twirl"]

    improvements = {m: [] for m in methods}

    # Compute improvement for each configuration
    for circuit_size in df["config.circuit_size"].unique():
        for noise_level in df["config.noise_level"].unique():
            subset = df[
                (df["config.circuit_size"] == circuit_size) &
                (df["config.noise_level"] == noise_level)
            ]

            none_row = subset[subset["config.mitigation"] == "none"]
            if none_row.empty or "result.absolute_error" not in none_row.columns:
                continue

            baseline_error = none_row.iloc[0].get("result.absolute_error", 0)
            if baseline_error <= 0:
                continue

            for method in methods:
                row = subset[subset["config.mitigation"] == method]
                if not row.empty and "result.absolute_error" in row.columns:
                    mitigated_error = row.iloc[0].get("result.absolute_error", baseline_error)
                    if mitigated_error > 0:
                        improvements[method].append(baseline_error / mitigated_error)

    # Plot mean improvement with error bars
    x = np.arange(len(methods))
    means = [np.mean(improvements[m]) if improvements[m] else 1.0 for m in methods]
    stds = [np.std(improvements[m]) if len(improvements[m]) > 1 else 0.0 for m in methods]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=[colors()[0], colors()[2]], edgecolor="black")

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="No improvement")
    ax.set_xlabel("Mitigation Method")
    ax.set_ylabel("Improvement Factor\n(higher is better)")
    ax.set_title(r"\textbf{(b) Improvement Factor}")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate values
    for bar, mean in zip(bars, means):
        ax.annotate(
            f"{mean:.1f}×",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )


def plot_error_vs_noise(ax, df: pd.DataFrame):
    """Panel (c): Error vs noise level for each method."""
    methods = ["none", "zne", "twirl"]
    method_labels = ["Unmitigated", "ZNE", "Twirl"]
    method_colors = [colors()[1], colors()[0], colors()[2]]

    noise_levels = sorted(df["config.noise_level"].unique())

    # Use median circuit size
    circuit_sizes = sorted(df["config.circuit_size"].unique())
    circuit_size = circuit_sizes[len(circuit_sizes) // 2] if circuit_sizes else 6

    for method, label, color in zip(methods, method_labels, method_colors):
        errors = []
        for noise in noise_levels:
            row = df[
                (df["config.circuit_size"] == circuit_size) &
                (df["config.noise_level"] == noise) &
                (df["config.mitigation"] == method)
            ]
            if not row.empty and "result.absolute_error" in row.columns:
                errors.append(row.iloc[0].get("result.absolute_error", np.nan))
            else:
                errors.append(np.nan)

        ax.plot(noise_levels, errors, "o-", color=color, label=label, markersize=6)

    ax.set_xlabel("Noise Level (depolarizing prob.)")
    ax.set_ylabel("Absolute Error")
    ax.set_title(rf"\textbf{{(c) Error vs Noise ({circuit_size}q)}}")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)


@bk.pplot
def plot_mitigation_effectiveness(df: pd.DataFrame):
    """Create error mitigation effectiveness figure."""
    fig, axes = plt.subplots(1, 3, figsize=(double_column_width(), 1.5))

    plot_error_comparison(axes[0], df)
    plot_improvement_factor(axes[1], df)
    plot_error_vs_noise(axes[2], df)

    return fig


if __name__ == "__main__":
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"

    # Load data
    data = bk.load_log("logs/error_mitigation/effectiveness.jsonl")

    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.json_normalize(data) if data else pd.DataFrame()

    if df.empty:
        print("No effectiveness data found")
        exit(1)

    print(f"Loaded {len(df)} rows")
    print(f"Circuit sizes: {sorted(df['config.circuit_size'].unique())}")
    print(f"Noise levels: {sorted(df['config.noise_level'].unique())}")
    print(f"Methods: {sorted(df['config.mitigation'].unique())}")

    fig = plot_mitigation_effectiveness(df)
    plt.tight_layout()
    plt.savefig("plots/mitigation_effectiveness.pdf", bbox_inches="tight")
    plt.show()
