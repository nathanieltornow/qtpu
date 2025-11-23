import matplotlib.pyplot as plt
import pandas as pd
from benchkit.plot.config import (
    PlotStyle,
    colors,
    double_column_width,
    register_style,
    single_column_width,
)

import benchkit as bk


def _plot_runtime_comparison(
    df: pd.DataFrame,
    ax: plt.Axes,
) -> None:
    run_df = df.rename(
        columns={
            "result.qac.runtime": "qac",
            "result.qtpu.runtime": "qtpu",
        },
    )
    bk.plot.line_comparison(
        ax, run_df, keys=["qac", "qtpu"], group_key="config.circuit_size", error=None
    )


def _plot_memory_comparison(
    df: pd.DataFrame,
    ax: plt.Axes,
) -> None:

    mem_df = df.rename(
        columns={
            "result.qac.generation_memory": "qac",
            "result.qtpu.generation_memory": "qtpu",
        },
    )
    bk.plot.line_comparison(
        ax,
        mem_df,
        keys=["qac", "qtpu"],
        group_key="config.circuit_size",
    )


FONTSIZE = 8

CUSTOM_RC = {
    "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "figure.titlesize": FONTSIZE,
}


@bk.pplot("scale", custom_rc=CUSTOM_RC)
def plot_scale_bench() -> None:
    df = bk.logging.join_logs(["logs/01_scale_qac.jsonl", "logs/01_scale_qtpu.jsonl"])
    df = df[(df["config.num_samples"] == 10000) | (df["config.num_samples"].isna())]

    # Compute runtimes
    df["result.qac.runtime"] = (
        df["result.qac.generation_time"]
        + df["result.qac.quantum_time"]
        + df["result.qac.classical_time"]
    )
    df["result.qtpu.runtime"] = (
        df["result.qtpu.generation_time"]
        + df["result.qtpu.quantum_time"]
        + df["result.qtpu.classical_time"]
    )

    # Memory to MB
    df["result.qac.generation_memory"] /= 1e6
    df["result.qtpu.generation_memory"] /= 1e6

    df = df[df["config.circuit_size"] > 10]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(double_column_width(), 1.7),
        sharex="col",
        sharey=False,  # Disable global sharey, we will enable row-wise manually
    )

    # Row 1: Runtime
    ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]
    ax2.sharey(ax1)
    ax4.sharey(ax3)
    # Split into qnn / wstate
    qnn_df = df[df["config.bench"] == "qnn"]
    ws_df = df[df["config.bench"] == "wstate"]

    # ----- RUNTIME PLOTS -----
    _plot_runtime_comparison(qnn_df, ax1)
    ax1.set_title("Runtime (qnn)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Circuit size")

    _plot_runtime_comparison(ws_df, ax2)
    ax2.set_title("Runtime (wstate)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Circuit size")

    # ----- MEMORY PLOTS -----
    _plot_memory_comparison(qnn_df, ax3)
    ax3.set_title("Memory (qnn)")
    ax3.set_yscale("log")
    ax3.set_xlabel("Circuit size")

    _plot_memory_comparison(ws_df, ax4)
    ax4.set_title("Memory (wstate)")
    ax4.set_yscale("log")
    ax4.set_xlabel("Circuit size")

    # One shared legend above all subplots
    handles, labels = ax1.get_legend_handles_labels()
    # BOTTOM of the figure
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.1),
    )

    # fig.tight_layout(rect=[0, 0, 1, 0.92])  # leave space for legend
    return fig


register_style(
    "qac", PlotStyle(hatch="//", sort_order=1, color=colors()[1], marker="s")
)
register_style(
    "qtpu",
    PlotStyle(hatch="\\\\", sort_order=0, color=colors()[0], marker="o"),
)

plot_scale_bench()
