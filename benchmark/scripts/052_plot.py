import pandas as pd
from benchmark._plot_util import *
from matplotlib.ticker import FixedLocator


data_path = "benchmark/results/sampling3.json"


def plot_end2end_2(ax):
    df = pd.read_json("benchmark/results/end_to_end_estimator.json")
    # df = df[df["name"] != "qaoa1"]

    # df["num_qubits"] = df["num_qubits"] // 20 * 20
    df = df[df["num_qubits"] <= 200]

    df = df[(df["m"] == 2) | (df["reps"] == 2)]
    # df.drop(columns=["name"], inplace=True)

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    df["qtpu"] = df["qtpu_gpu_run"] + df["qtpu_gpu_post"]
    df["cutensor"] = df["cutensor_exec"] + df["cutensor_compile"]
    for name, linestyle in zip(["vqe", "qml", "qaoa1", "qaoa2"], linestyles):
        df_mean = (
            df[df["name"] == name].drop(columns=["name"]).groupby(["num_qubits"]).mean()
        )
        df_std = (
            df[df["name"] == name].drop(columns=["name"]).groupby(["num_qubits"]).std()
        )

        df_mean.plot.line(
            y=["qtpu"],
            yerr=df_std,
            ax=ax,
            linewidth=2,
            capsize=3,
            marker="o",
            linestyle=linestyle,
            legend=False,
        )

    df = df[df["num_qubits"] % 20 == 0]
    df_mean = df.drop(columns=["name"]).groupby(["num_qubits"]).mean()
    df_std = df.drop(columns=["name"]).groupby(["num_qubits"]).std()
    df_mean.plot.line(
        y=["cutensor"],
        yerr=df_std,
        ax=ax,
        linewidth=2,
        capsize=3,
        marker="v",
        color="black",
        legend=False,
    )

    # df_mean.plot.line(
    #     y=["qtpu_gpu_post"],
    #     yerr=df_std,
    #     ax=ax,
    #     linewidth=2,
    #     capsize=3,
    #     marker="v",
    #     color="black",
    #     legend=False,
    # )

    df["qtpu_gpu_run_percents"] = df["qtpu_gpu_run"] / df["qtpu"] * 100
    df_mean = df.drop(columns=["name"]).mean()
    print(df_mean)

    ax.set_yscale("log")
    ax.set_ylabel("Runtime [s]")
    ax.set_xlabel("Number of qubits")


def generate_figure():

    fig, (ax, ax0, ax1) = plt.subplots(1, 3, figsize=(15, 1.9))
    fig.subplots_adjust(wspace=0.2)

    df = pd.read_json(data_path)
    # df = df[df["name"] == "qaoa2"]
    df = df[df["num_samples"].isin([100, 500, 1000, 5000, 10000])]

    df["cutensor"] = df["cutensor_exec"] + df["cutensor_compile"]

    df["qtpu"] = df["qtpu_gpu_run"] + df["qtpu_gpu_post"]
    df["ckt"] = df["ckt_run"] + df["ckt_post"]




    df["qtpu_rel_post"] = df["qtpu_gpu_post"] / df["qtpu"]
    df["ckt_rel_post"] = df["ckt_post"] / df["ckt"]

    df.drop(columns=["name"], inplace=True)

    df_mean = df.groupby(["num_samples"]).mean()
    df_std = df.groupby(["num_samples"]).std()


    average_speedup = df_mean["ckt"].mean() / df_mean["qtpu"].mean()
    max_speedup = df_mean["ckt"].max() / df_mean["qtpu"].max()
    print(average_speedup, max_speedup)



    average_rel_ckt = df_mean["ckt_rel_post"].mean()
    max_rel_ckt = df_mean["ckt_rel_post"].max()
    average_rel_qtpu = df_mean["qtpu_rel_post"].mean() * 100
    max_rel_qtpu = df_mean["qtpu_rel_post"].max() * 100
    # average_rel_qtpu = df_mean["qtpu_rel_post"].mean()

    print("average_rel_ckt", average_rel_ckt)
    print("max_rel_ckt", max_rel_ckt)
    print("average_rel_qtpu", average_rel_qtpu)
    print("max_rel_qtpu", max_rel_qtpu)

    mean_cutensor = df_mean["cutensor"].mean()
    print(mean_cutensor)

    df_mean.plot.bar(
        y=["qtpu", "ckt"],
        yerr=df_std,
        legend=False,
        rot=0,
        width=0.8,
        edgecolor="black",
        linewidth=2,
        ax=ax0,
        capsize=3,
    )

    ax0.axhline(
        mean_cutensor, color="black", linestyle="--", linewidth=2, label="cutensor"
    )
    ax1.axhline(0.5, color="blue", linestyle="--", linewidth=2, label="50%")

    ax.sharey(ax0)
    df_mean.plot.bar(
        y=["qtpu_rel_post", "ckt_rel_post"],
        yerr=df_std,
        legend=False,
        rot=0,
        width=0.8,
        edgecolor="black",
        linewidth=2,
        ax=ax1,
        capsize=3,
    )

    plot_end2end_2(ax)

    # ax1.axhline(1, color="black", linestyle="--", linewidth=2)

    # ax0.set_yscale("log")
    ax1.set_yscale("log")
    postprocess_barplot(ax0)
    postprocess_barplot(ax1)

    ax0.set_ylabel("Runtime [s]")
    ax1.set_ylabel("Rel. postproc. time")

    ax0.legend(["cuTensorNet", "QTPU", "QAC"], loc="upper left")
    ax1.legend(["50%", "QTPU", "QAC"], loc="upper left")

    ax.set_title(
        "(a) End-to-end runtime vs. cuTensorNet", fontweight="bold", color="black"
    )
    ax0.set_title("(b) End-to-end runtime vs. QAC (40q)", fontweight="bold", color="black")
    ax1.set_title("(c) Postprocessing time / total runtime (40q)", fontweight="bold", color="black")

    ax0.set_xlabel("Number of QPD-samples")
    ax1.set_xlabel("Number of QPD-samples")

    ax.legend(
        ["VQE", "QML", "QAOA I", "QAOA II", "CuTensorNet"],
    )

    # fig.text(

    #     0.3,
    #     1.04,
    #     "Higher is better ↑",
    #     ha="center",
    #     va="center",
    #     fontweight="bold",
    #     color="midnightblue",
    # )

    fig.text(
        0.5,
        1.04,
        "Lower is better ↓",
        ha="center",
        va="center",
        fontweight="bold",
        color="midnightblue",
    )

    fig.savefig("benchmark/plots/05_sampling.pdf", bbox_inches="tight")


generate_figure()
