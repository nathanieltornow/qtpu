"""Plotting for Real Hardware Benchmark."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 12, "figure.dpi": 150})


def load_hardware_data(log_path: str) -> pd.DataFrame:
    df = pd.read_json(log_path, lines=True)
    df["circuit_size"] = df["config"].apply(lambda x: x["circuit_size"])
    for col in [
        "estimated_qpu_time",
        "actual_qpu_time",
        "estimation_error",
        "mean_fidelity",
        "min_fidelity",
        "num_subcircuits",
        "compile_time",
    ]:
        df[col] = df["result"].apply(lambda x: x.get(col) if x else None)
    return df.dropna(subset=["mean_fidelity"])


def plot_time_validation():
    """Plot estimated vs actual QPU time."""
    df = load_hardware_data("logs/hardware/qnn_hardware.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Scatter: estimated vs actual
    ax = axes[0]
    ax.scatter(
        df["estimated_qpu_time"],
        df["actual_qpu_time"],
        c=df["circuit_size"],
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidth=0.5,
    )
    lims = [
        0,
        max(df["estimated_qpu_time"].max(), df["actual_qpu_time"].max()) * 1.1,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")
    ax.set_xlabel("Estimated QPU time (s)")
    ax.set_ylabel("Actual QPU time (s)")
    ax.set_title("(a) QPU Time Validation")
    ax.legend()
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Circuit size (qubits)")

    # (b) Estimation error by circuit size
    ax = axes[1]
    sizes = sorted(df["circuit_size"].unique())
    errors = [
        df[df["circuit_size"] == s]["estimation_error"].values[0] * 100
        for s in sizes
    ]
    ax.bar(range(len(sizes)), errors, color="#1f77b4")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Circuit size (qubits)")
    ax.set_ylabel("Estimation error (%)")
    ax.set_title("(b) Time Estimation Accuracy")

    plt.tight_layout()
    plt.savefig("plots/hardware/time_validation.pdf", bbox_inches="tight")
    plt.savefig("plots/hardware/time_validation.png", bbox_inches="tight")
    print("Saved plots/hardware/time_validation.{pdf,png}")


def plot_fidelity():
    """Plot fidelity vs circuit size."""
    df = load_hardware_data("logs/hardware/qnn_hardware.jsonl")

    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = sorted(df["circuit_size"].unique())
    mean_fids = [
        df[df["circuit_size"] == s]["mean_fidelity"].values[0] for s in sizes
    ]
    min_fids = [
        df[df["circuit_size"] == s]["min_fidelity"].values[0] for s in sizes
    ]

    x = np.arange(len(sizes))
    ax.bar(x, mean_fids, color="#1f77b4", label="Mean fidelity")
    ax.scatter(x, min_fids, color="#d62728", zorder=5, label="Min fidelity", marker="v")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Circuit size (qubits)")
    ax.set_ylabel("Hellinger fidelity")
    ax.set_title("Subcircuit Fidelity on IBM Marrakesh")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/hardware/fidelity.pdf", bbox_inches="tight")
    plt.savefig("plots/hardware/fidelity.png", bbox_inches="tight")
    print("Saved plots/hardware/fidelity.{pdf,png}")


if __name__ == "__main__":
    import os

    os.makedirs("plots/hardware", exist_ok=True)
    plot_time_validation()
    plot_fidelity()
