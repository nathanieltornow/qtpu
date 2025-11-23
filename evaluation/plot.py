import benchkit as bk

import pandas as pd
import matplotlib.pyplot as plt


from benchkit.plot.config import (
    register_style,
    PlotStyle,
    single_column_width,
    double_column_width,
    colors,
)


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
    bk.plot.bar_comparison(
        ax,
        run_df,
        keys=["qac", "qtpu"],
        group_key="config.circuit_size",
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
    bk.plot.bar_comparison(
        ax,
        mem_df,
        keys=["qac", "qtpu"],
        group_key="config.circuit_size",
    )


@bk.pplot("scale")
def plot_scale_bench() -> None:
    df = bk.load_log("logs/01_scale.jsonl")

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

    # One single row of 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1,
        4,
        figsize=(double_column_width(), 1.7),  # wide figure, short height
        sharey=False,
    )

    # Split into qnn / wstate
    qnn_df = df[df["config.bench"] == "qnn"]
    ws_df = df[df["config.bench"] == "wstate"]

    # ----- RUNTIME PLOTS -----
    _plot_runtime_comparison(qnn_df, ax1)
    ax1.set_title("Runtime (QNN)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Circuit size")

    _plot_runtime_comparison(ws_df, ax2)
    ax2.set_title("Runtime (Wstate)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Circuit size")

    # ----- MEMORY PLOTS -----
    _plot_memory_comparison(qnn_df, ax3)
    ax3.set_title("Memory (QNN)")
    ax3.set_yscale("log")
    ax3.set_xlabel("Circuit size")

    _plot_memory_comparison(ws_df, ax4)
    ax4.set_title("Memory (Wstate)")
    ax4.set_yscale("log")
    ax4.set_xlabel("Circuit size")

    # One shared legend above all subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.92])  # leave space for legend
    return fig


register_style("qac", PlotStyle(hatch="//", sort_order=1, color=colors()[1]))
register_style(
    "qtpu",
    PlotStyle(hatch="\\\\", sort_order=0, color=colors()[0]),
)

plot_scale_bench()
