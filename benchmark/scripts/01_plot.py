import pandas as pd
from benchmark._plot_util import *
from matplotlib.ticker import FixedLocator
import numpy as np


data_path = "benchmark/results/postprocessing.json"
FIG_SIZE = (15, 1.9)


def get_dfs():
    df = pd.read_json(data_path)
    df = df[df["num_qubits"] <= 100]

    dfs = {
        "VQE": df[df["name"] == "vqe"].copy().drop(columns=["name"]),
        "QML": df[df["name"] == "qml"].copy().drop(columns=["name"]),
        "QAOA I": df[df["name"] == "qaoa1"].copy().drop(columns=["name"]),
        "QAOA II": df[df["name"] == "qaoa2"].copy().drop(columns=["name"]),
    }
    return dfs


def generate_overhead_figure():
    dfs = get_dfs()
    fig, axes = plt.subplots(1, len(dfs), figsize=FIG_SIZE, sharey=True)
    fig.subplots_adjust(wspace=0.1)
    # axes[0].set_yscale("log")

    for i, (title, df) in enumerate(dfs.items()):
        # df["qtpu_cost"] = 10 ** df["qtpu_cost_log10"]
        # df["ckt_cost"] = 10 ** df["ckt_cost_log10"]
        df["qtpu_cost"] = df["qtpu_cost_log10"] + (
            df["num_qpds"] * np.log(9) / np.log(10)
        )
        df["ckt_cost"] = df["ckt_cost_log10"] + (
            df["num_qpds"] * np.log(9) / np.log(10)
        )
        df_mean = df.groupby(["num_qubits"]).mean()
        df_std = df.groupby(["num_qubits"]).std()

        df_mean.plot.bar(
            # x="num_qubits",
            y=["qtpu_cost", "ckt_cost"],
            yerr=df_std,
            legend=False,
            rot=0,
            width=0.8,
            edgecolor="black",
            linewidth=2,
            ax=axes[i],
            capsize=3,
        )
        postprocess_barplot(axes[i])
        axes[i].legend(["QTPU", "QAC"], loc="upper left")
        axes[i].set_title(f"{chr(97 + i)}) {title}", fontweight="bold", color="black")
        axes[i].set_ylabel("Postproc. overhead [FLOPs]")
        axes[i].set_xlabel("Number of qubits")

        # transform y-axis ticks to 10^x
        yticks = axes[i].get_yticks()
        axes[i].yaxis.set_major_locator(FixedLocator(yticks))
        axes[i].set_yticklabels([f"$10^{{{int(y)}}}$" for y in yticks])

    fig.text(
        0.5,
        0.95,
        "Lower is better ↓",
        ha="center",
        va="center",
        fontweight="bold",
        color="midnightblue",
    )
    fig.savefig("benchmark/plots/01_post_overhead.pdf", bbox_inches="tight")


def generate_speedup_figure():
    dfs = get_dfs()
    fig, axes = plt.subplots(1, len(dfs), figsize=FIG_SIZE, sharey=True)
    fig.subplots_adjust(wspace=0.1)
    axes[0].set_yscale("log")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [colors[0], colors[2], colors[1]]

    for i, (title, df) in enumerate(dfs.items()):
        df_mean = df.groupby(["num_qubits"]).mean()
        df_std = df.groupby(["num_qubits"]).std()

        df_mean.plot.bar(
            # x="num_qubits",
            y=["qtpu_post", "qtpu_gpu_post", "ckt_post"],
            legend=False,
            rot=0,
            width=0.8,
            edgecolor="black",
            linewidth=2,
            ax=axes[i],
            color=colors,
            yerr=df_std,
            capsize=3,
        )
        postprocess_barplot(axes[i], ["//", "oo", "\\\\"])
        axes[i].legend(["QTPU", "QTPU (GPU)", "QAC"], loc="upper left")
        axes[i].set_title(f"{chr(97 + i)}) {title}", fontweight="bold", color="black")
        axes[i].set_ylabel("Postprocessing time [s]")
        axes[i].set_xlabel("Number of qubits")

    fig.text(
        0.5,
        0.95,
        "Lower is better ↓",
        ha="center",
        va="center",
        fontweight="bold",
        color="midnightblue",
    )

    fig.savefig("benchmark/plots/01_post_speedup.pdf", bbox_inches="tight")


def print_summary():
    df = pd.read_json(data_path)

    # Convert log10 costs to actual costs
    df["qtpu_cost"] = 10 ** df["qtpu_cost_log10"]
    df["ckt_cost"] = 10 ** df["ckt_cost_log10"]

    # Calculate the speedup
    df["overhead_speedup"] = df["ckt_cost"] / df["qtpu_cost"]
    df["postprocessing_speedup"] = df["ckt_post"] / df["qtpu_post"]
    df["gpu_speedup"] = df["qtpu_post"] / df["qtpu_gpu_post"]

    # print(
    #     f"Overhead speedup: {df['overhead_speedup'].mean()}, {df['overhead_speedup'].max()}"
    # )

    # print(
    #     f"Postprocessing speedup: {df['postprocessing_speedup'].mean()}, {df['postprocessing_speedup'].max()}"
    # )
    # print(f"GPU speedup: {df['gpu_speedup'].mean()}, {df['gpu_speedup'].max()}")

    df.drop(columns=["name"], inplace=True)
    print(df.groupby(["num_qubits"]).mean().to_json(indent=4))
    df = df[
        [
            "qtpu_cost",
            "ckt_cost",
            "ckt_post",
            "qtpu_post",
            "qtpu_gpu_post",
            "overhead_speedup",
            "postprocessing_speedup",
            "gpu_speedup",
        ]
    ]

    print(df.aggregate(["mean", "std", "min", "max"]).to_json(indent=4))


generate_overhead_figure()
generate_speedup_figure()

print_summary()
