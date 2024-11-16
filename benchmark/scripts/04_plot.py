import pandas as pd

from benchmark._plot_util import *
import numpy as np
from scipy.optimize import curve_fit


data_path = "benchmark/results/threshold2.json"


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def generate_figure():
    df = pd.read_json(data_path)
    df["cutensor"] = df["cutensor_exec"] + df["cutensor_compile"]
    df["ckt_diff"] = df["cutensor"] - df["ckt_post"]
    df["qtpu_diff"] = df["cutensor"] - df["qtpu_gpu_post"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 1.9), sharey=True)
    fig.subplots_adjust(wspace=0.1)

    bottom = 0.1
    axes[0].set_ylim(bottom=bottom, top=5e2)

    print(df["qtpu_diff"].min(), df["ckt_diff"].min())
    print(df["qtpu_diff"].max(), df["ckt_diff"].max())


    df["ckt_rel"] = df["ckt_post"] / df["cutensor"]
    print(df["ckt_rel"].mean())

    for ax, r in zip(axes, [4, 5]):

        df_r = df[df["r"] == r].drop(columns=["name"])

        # put df_r["ckt_diff"] = y for all  df_r["ckt_diff"] < 0
        df_r.loc[df_r["ckt_diff"] < 0, "ckt_diff"] = ax.get_ylim()[0] * 0.1

        df_mean = df_r.groupby("n").mean()
        df_std = df_r.groupby("n").std()

        df_mean.plot.bar(
            y=["qtpu_diff", "ckt_diff"],
            yerr=df_std,
            legend=False,
            rot=0,
            width=0.8,
            edgecolor="black",
            linewidth=2,
            ax=ax,
            capsize=3,
        )

        ax.set_yscale("log")

        #put an "x" for all bars with y == 0.01
        for bar in ax.patches:
            if bar.get_height() < bottom:
                # get location of the bar
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.text(
                    x,
                    bottom,
                    "X",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontweight="bold",
                )

            ax.set_xlabel("QAOA cluster size (# qubits)")

        

        postprocess_barplot(ax)
        ax.legend(["QTPU", "QAC"], loc="upper left")
    df.drop(columns=["name"], inplace=True)

    df["ckt_diff"] = df["cutensor"] - df["ckt_post"]
    df["qtpu_diff"] = df["cutensor"] - df["qtpu_gpu_post"]

    print(df[df["r"] == 5].groupby("n")["cutensor"].mean())

    df = df[["r", "ckt_diff", "qtpu_diff"]]
    print(df.groupby("r").aggregate(["min", "max"]).to_json(indent=4))

    # axes[0].set_ylim(top=1e3)
    axes[0].set_title("(a) QPU-time threshold (4 clusters)", fontweight="bold")
    axes[1].set_title("(b) QPU-time threshold (5 clusters)", fontweight="bold")


    axes[0].set_ylabel("QPU-time threshold [s]")

    # extrapolate "cutensor" to n=[22, 23, 24, 25]

    fig.text(
        0.5,
        1.05,
        "Higher is better ↑",
        ha="center",
        va="center",
        fontweight="bold",
        color="midnightblue",
    )
    # fig.savefig("benchmark/plots/04_threshold.pdf", bbox_inches="tight")
    fig.savefig("benchmark/plots/04_threshold.png", bbox_inches="tight", dpi=400)


generate_figure()
