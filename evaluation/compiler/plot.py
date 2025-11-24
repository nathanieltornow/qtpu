import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import benchkit as bk
from benchkit.plot.config import (
    get_style,
    register_style,
    PlotStyle,
    single_column_width,
    colors,
)

from evaluation.analysis import base_line_df


@bk.pplot
def plot_compile_time(qtpu_df, qac_df):

    # merge on circuit_size
    # qtpu_df = qtpu_df[qtpu_df["config.num_trials"] == 10]
    merged_df = pd.merge(
        qtpu_df,
        qac_df,
        on=["config.circuit_size", "config.bench"],
        suffixes=(".qtpu", ".qac"),
    )
    merged_df.rename(
        columns={"result.qtpu.compile_time": "qtpu", "result.qac.compile_time": "qac"},
        inplace=True,
    )
    merged_df = merged_df[merged_df["config.max_qubits"] == 30]
    merged_df = merged_df[merged_df["config.num_trials"] == 100]
    merged_df = merged_df[merged_df["config.bench"] == "vqe_su2"]
    merged_df = merged_df[merged_df["result.qtpu.qtensor_widths"].apply(max) <= 30]

    merged_df = merged_df[
        merged_df["config.circuit_size"].isin(list(range(10, 110, 30)))
    ]

    fig, ax = plt.subplots(figsize=(single_column_width(), 1.7))
    bk.plot.bar_comparison(
        ax=ax,
        results=merged_df,
        keys=["qtpu", "qac"],
        group_key="config.circuit_size",
        error="std",
    )
    return fig


@bk.pplot
def plot_qcost(qtpu_df, qac_df):
    fig, ax = plt.subplots(figsize=(single_column_width(), 1.7))

    # merge on circuit_size
    merged_df = pd.merge(
        qtpu_df,
        qac_df,
        on=["config.circuit_size", "config.bench"],
        suffixes=(".qtpu", ".qac"),
    )

    merged_df = base_line_df(merged_df)

    merged_df["qtpu.q_cost"] = merged_df["result.qtpu.qtensor_errors"].apply(max)
    merged_df["qac.q_cost"] = merged_df["result.qac.qtensor_errors"].apply(max)

    merged_df["qtpu.delta_q_cost"] = (
        merged_df["qtpu.q_cost"] - merged_df["baseline.error"]
    )
    merged_df["qac.delta_q_cost"] = (
        merged_df["qac.q_cost"] - merged_df["baseline.error"]
    )
    merged_df["qtpu.delta_c_cost"] = np.log(merged_df["result.qtpu.c_cost"] + 1)

    merged_df["qac.delta_c_cost"] = np.log(merged_df["result.qac.c_cost"] + 1)

    merged_df["qtpu"] = merged_df["qtpu.delta_q_cost"] / merged_df["qtpu.delta_c_cost"]
    merged_df["qac"] = merged_df["qac.delta_q_cost"] / merged_df["qac.delta_c_cost"]

    merged_df["qtpu"] = merged_df["qtpu"].replace([np.inf, -np.inf], np.nan)
    merged_df["qac"] = merged_df["qac"].replace([np.inf, -np.inf], np.nan)

    merged_df = merged_df.dropna(subset=["qtpu", "qac"])
    bk.plot.bar_comparison(
        ax=ax,
        results=merged_df,
        keys=["qtpu", "qac"],
        # keys=["result.qtpu.c_cost", "result.qac.c_cost"],
        group_key="config.circuit_size",
        error="std",
    )
    # ax.set_yscale("log")

    return fig


# ------------------------------------------------------------
# Run manually
# ------------------------------------------------------------
if __name__ == "__main__":
    register_style("qtpu", PlotStyle(marker="o", color=colors()[0]))
    register_style("qac", PlotStyle(marker="s", color=colors()[1]))

    qtpu_df = bk.load_log("logs/compile/qtpu.jsonl")
    qac_df = bk.load_log("logs/compile/qac.jsonl")

    # plot_compiler_results(qtpu_df, qac_df)
    plot_compile_time(qtpu_df, qac_df)
    plot_qcost(qtpu_df, qac_df)
