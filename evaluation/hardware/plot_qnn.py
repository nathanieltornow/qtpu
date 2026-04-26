"""QNN fidelity plots: IBM Marrakesh (HW) + Pauli-noise sim tail + Pareto frontier."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchkit.plot.config import (
    PlotStyle,
    colors,
    register_style,
    double_column_width,
)

import benchkit as bk

QTPU_LABEL = r"\textsc{qTPU}"
BASELINE_LABEL = r"\textsc{Baseline}"
HW_LABEL = r"\textsc{Marrakesh} (40q)"
SIM_LABEL = r"\textsc{Pauli-Sim} (80q)"

HW_LOG = "logs/hardware/qnn_marrakesh.jsonl"
SIM_LOG = "logs/hardware/qnn_sim_sweep.jsonl"
PARETO_HW_LOG = "logs/hardware/pareto_40q.jsonl"
PARETO_SIM_LOG = "logs/hardware/sim_pareto_80q.jsonl"


def _load_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _plot_bars(ax, sizes, qtpu_vals, base_vals, title, *, show_ylabel, show_legend):
    x = np.arange(len(sizes))
    width = 0.4
    ax.bar(x - width / 2, qtpu_vals, width, label=QTPU_LABEL,
           color=colors()[0], edgecolor="black", linewidth=1, hatch="//")
    ax.bar(x + width / 2, base_vals, width, label=BASELINE_LABEL,
           color=colors()[1], edgecolor="black", linewidth=1, hatch="\\\\")
    ax.set_xlabel("Circuit Size", labelpad=2)
    if show_ylabel:
        ax.set_ylabel("Fidelity\n(higher is better)")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    if show_legend:
        ax.legend(loc="upper right", handlelength=1.2, handletextpad=0.4, borderaxespad=0.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}q" for s in sizes], rotation=30, ha="right", rotation_mode="anchor")


def plot_hw(ax, hw_df, *, show_ylabel, show_legend):
    sizes = sorted(int(n) for n in hw_df["n"].unique())
    qtpu_vals, base_vals = [], []
    for s in sizes:
        row = hw_df[hw_df["n"] == s].iloc[0]
        qtpu_vals.append(float(row["cut"]["expval"]))
        base_vals.append(float(row["mono"]["expval"]))
    _plot_bars(ax, sizes, qtpu_vals, base_vals,
               r"\textbf{(a) Real Hardware (IBM Marrakesh)}",
               show_ylabel=show_ylabel, show_legend=show_legend)


def plot_sim(ax, sim_df, *, show_ylabel, show_legend):
    sizes = sorted(int(n) for n in sim_df["n"].unique())
    qtpu_vals, base_vals = [], []
    for s in sizes:
        row = sim_df[sim_df["n"] == s].iloc[0]
        qtpu_vals.append(float(row["cut"]))
        base_vals.append(float(row["mono_mean"]))
    _plot_bars(ax, sizes, qtpu_vals, base_vals,
               r"\textbf{(b) Noise-Model Simulation}",
               show_ylabel=show_ylabel, show_legend=show_legend)


def plot_pareto(ax, hw_df, sim_df, *, show_ylabel, show_legend):
    """Pareto frontier: fidelity vs classical cost under Pauli-noise sim."""
    sim_cut = sim_df[sim_df["mode"] == "cut"].sort_values("c_cost")
    sim_mono = sim_df[sim_df["mode"] == "monolithic"].iloc[0]

    # Use c_cost+1 to keep mono (c_cost=0) on a log axis.
    sim_x = sim_cut["c_cost"].to_numpy()
    sim_y = 1.0 - sim_cut["fidelity_mean"].to_numpy()
    mono_err = 1.0 - float(sim_mono["fidelity_mean"])

    # Full chain from Baseline (c_cost=0) through all qTPU Pareto points.
    chain_x = np.concatenate([[0], sim_x])
    chain_y = np.concatenate([[mono_err], sim_y])

    # Connecting line + filled area for the whole Pareto chain.
    ax.plot(chain_x, chain_y, "-", color=colors()[0], linewidth=1.5, zorder=3)
    ax.fill_between(chain_x, chain_y, alpha=0.2, color=colors()[0], zorder=1)
    ax.scatter(chain_x, chain_y, marker="o", s=45, color=colors()[0],
               edgecolors="black", linewidths=0.8, zorder=4)

    ax.set_xlabel("Classical Cost [FLOPs]", labelpad=2)
    if show_ylabel:
        ax.set_ylabel("Error\n(lower is better)")
    ax.set_title(r"\textbf{(c) Pareto Frontier (Noise-Model Sim)}")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)


@bk.pplot
def plot_qnn(hw_df, sim_df, pareto_hw_df, pareto_sim_df):
    """3-panel figure: HW fidelity-vs-size, sim fidelity-vs-size, HW+sim Pareto."""
    fig, axes = plt.subplots(
        1, 3, figsize=(double_column_width(), 1.5),
        gridspec_kw={"width_ratios": [1, 1, 1.1]},
    )
    plot_hw(axes[0], hw_df, show_ylabel=True, show_legend=True)
    plot_sim(axes[1], sim_df, show_ylabel=False, show_legend=False)
    plot_pareto(axes[2], pareto_hw_df, pareto_sim_df,
                show_ylabel=True, show_legend=True)
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylim(0, 1.0)
    axes[2].set_ylim(0, 1.0)
    axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(wspace=0.08)
    fig.tight_layout()
    # Shift panel (c) rightward to create a visual gap from (a)+(b).
    # Done after tight_layout; neutralize pplot's later tight_layout call.
    pos = axes[2].get_position()
    axes[2].set_position([pos.x0 + 0.035, pos.y0, pos.width, pos.height])
    fig.tight_layout = lambda *a, **k: None
    return fig


if __name__ == "__main__":
    register_style("qtpu", PlotStyle(color=colors()[0], hatch="//"))
    register_style("baseline", PlotStyle(color=colors()[1], hatch="\\\\"))

    hw_df = _load_jsonl(HW_LOG)
    sim_df = _load_jsonl(SIM_LOG)
    pareto_hw_df = _load_jsonl(PARETO_HW_LOG)
    pareto_sim_df = _load_jsonl(PARETO_SIM_LOG)

    for name, df in [("HW", hw_df), ("sim", sim_df),
                     ("pareto HW", pareto_hw_df), ("pareto sim", pareto_sim_df)]:
        if df.empty:
            print(f"No {name} data")
            exit(1)

    print(f"HW: {len(hw_df)} rows, sizes={sorted(hw_df['n'].unique())}")
    print(f"Sim: {len(sim_df)} rows, sizes={sorted(sim_df['n'].unique())}")
    print(f"Pareto HW: {len(pareto_hw_df)} rows")
    print(f"Pareto sim: {len(pareto_sim_df)} rows")
    plot_qnn(hw_df, sim_df, pareto_hw_df, pareto_sim_df)
