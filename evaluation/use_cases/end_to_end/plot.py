"""Plotting for End-to-End Composability Benchmark."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 12, "figure.dpi": 150})

QTPU_LOG = "logs/end_to_end/qtpu.jsonl"
BASELINE_LOG = "logs/end_to_end/baseline.jsonl"


def load_data():
    qtpu_df = pd.read_json(QTPU_LOG, lines=True)
    baseline_df = pd.read_json(BASELINE_LOG, lines=True)
    return qtpu_df, baseline_df


def plot_e2e_comparison():
    """Plot end-to-end comparison: wall time, circuits, code lines."""
    qtpu_df, baseline_df = load_data()

    # Extract config and result fields
    qtpu_df["circuit_size"] = qtpu_df["config"].apply(lambda x: x["circuit_size"])
    baseline_df["circuit_size"] = baseline_df["config"].apply(
        lambda x: x["circuit_size"]
    )

    for col in [
        "compile_time",
        "quantum_time",
        "total_time",
        "num_circuits",
        "total_code_lines",
    ]:
        qtpu_df[col] = qtpu_df["result"].apply(lambda x: x.get(col) if x else None)
        baseline_df[col] = baseline_df["result"].apply(
            lambda x: x.get(col) if x else None
        )

    # Drop rows where result is None
    qtpu_df = qtpu_df.dropna(subset=["total_time"])
    baseline_df = baseline_df.dropna(subset=["total_time"])

    sizes = sorted(qtpu_df["circuit_size"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) End-to-end wall time
    ax = axes[0]
    qtpu_times = [
        qtpu_df[qtpu_df["circuit_size"] == s]["total_time"].values[0] for s in sizes
    ]
    base_times = [
        baseline_df[baseline_df["circuit_size"] == s]["total_time"].values[0]
        for s in sizes
    ]
    x = np.arange(len(sizes))
    w = 0.35
    ax.bar(x - w / 2, base_times, w, label="Baseline", color="#d62728")
    ax.bar(x + w / 2, qtpu_times, w, label="qTPU", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Circuit size (qubits)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("(a) End-to-end wall time")
    ax.set_yscale("log")
    ax.legend()

    # (b) Number of circuits generated
    ax = axes[1]
    qtpu_circs = [
        qtpu_df[qtpu_df["circuit_size"] == s]["num_circuits"].values[0] for s in sizes
    ]
    base_circs = [
        baseline_df[baseline_df["circuit_size"] == s]["num_circuits"].values[0]
        for s in sizes
    ]
    ax.bar(x - w / 2, base_circs, w, label="Baseline", color="#d62728")
    ax.bar(x + w / 2, qtpu_circs, w, label="qTPU", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Circuit size (qubits)")
    ax.set_ylabel("Circuits generated")
    ax.set_title("(b) Circuit count")
    ax.set_yscale("log")
    ax.legend()

    # (c) Lines of generated code
    ax = axes[2]
    qtpu_loc = [
        qtpu_df[qtpu_df["circuit_size"] == s]["total_code_lines"].values[0]
        for s in sizes
    ]
    base_loc = [
        baseline_df[baseline_df["circuit_size"] == s]["total_code_lines"].values[0]
        for s in sizes
    ]
    ax.bar(x - w / 2, base_loc, w, label="Baseline", color="#d62728")
    ax.bar(x + w / 2, qtpu_loc, w, label="qTPU", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Circuit size (qubits)")
    ax.set_ylabel("Lines of code")
    ax.set_title("(c) Generated code size")
    ax.set_yscale("log")
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/end_to_end/e2e_comparison.pdf", bbox_inches="tight")
    plt.savefig("plots/end_to_end/e2e_comparison.png", bbox_inches="tight")
    print("Saved plots/end_to_end/e2e_comparison.{pdf,png}")


def plot_e2e_breakdown():
    """Plot qTPU timing breakdown: compile vs quantum vs classical."""
    qtpu_df, _ = load_data()

    qtpu_df["circuit_size"] = qtpu_df["config"].apply(lambda x: x["circuit_size"])
    for col in ["compile_time", "quantum_time", "classical_time"]:
        qtpu_df[col] = qtpu_df["result"].apply(lambda x: x.get(col) if x else None)
    qtpu_df = qtpu_df.dropna(subset=["compile_time"])

    sizes = sorted(qtpu_df["circuit_size"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    compile_times = [
        qtpu_df[qtpu_df["circuit_size"] == s]["compile_time"].values[0] for s in sizes
    ]
    quantum_times = [
        qtpu_df[qtpu_df["circuit_size"] == s]["quantum_time"].values[0] for s in sizes
    ]
    classical_times = [
        qtpu_df[qtpu_df["circuit_size"] == s]["classical_time"].values[0] for s in sizes
    ]

    x = np.arange(len(sizes))
    ax.bar(x, compile_times, label="Compilation", color="#ff7f0e")
    ax.bar(x, quantum_times, bottom=compile_times, label="Quantum (est.)", color="#1f77b4")
    bottoms = [c + q for c, q in zip(compile_times, quantum_times)]
    ax.bar(x, classical_times, bottom=bottoms, label="Classical", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Circuit size (qubits)")
    ax.set_ylabel("Time (s)")
    ax.set_title("qTPU End-to-End Timing Breakdown")
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/end_to_end/e2e_breakdown.pdf", bbox_inches="tight")
    plt.savefig("plots/end_to_end/e2e_breakdown.png", bbox_inches="tight")
    print("Saved plots/end_to_end/e2e_breakdown.{pdf,png}")


if __name__ == "__main__":
    import os

    os.makedirs("plots/end_to_end", exist_ok=True)
    plot_e2e_comparison()
    plot_e2e_breakdown()
