import pandas as pd
from benchmark._plot_util import *
from matplotlib.ticker import FixedLocator


data_path = "benchmark/results/sampling2.json"


def plot_end2end_2(ax):
    df = pd.read_json(data_path)
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

def generate_figure(axes, df_):

    df_["rel_error_ckt"] = abs(
        df_["ckt_res"] - df_["perf_res"]
    )  / abs(df["perf_res"])
    df_["rel_error_qtpu"] = abs(
        df_["qtpu_res"] - df_["perf_res"]
    )  / abs(df["perf_res"])

    df_["qtpu"] = df_["qtpu_gpu_run"] + df_["qtpu_gpu_post"]
    df_["ckt"] = df_["ckt_run"] + df_["ckt_post"]

    df_.drop(columns=["name"], inplace=True)

    df_mean = df_.groupby(["num_samples"]).mean()
    df_std = df_.groupby(["num_samples"]).std()

    df_mean.plot.bar(
        y=["rel_error_qtpu", "rel_error_ckt"],
        yerr=df_std,
        legend=False,
        rot=0,
        width=0.8,
        edgecolor="black",
        linewidth=2,
        ax=axes[0],
        capsize=3,
    )

    df_mean.plot.bar(
        y=["qtpu", "ckt"],
        yerr=df_std,
        legend=False,
        rot=0,
        width=0.8,
        edgecolor="black",
        linewidth=2,
        ax=axes[1],
        capsize=3,
    )

    postprocess_barplot(axes[0])
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Abs. error")
    axes[1].set_ylabel("Runime [s]")
    axes[0].set_title("Error")
    axes[1].set_title("Runtime")
    axes[0].set_xlabel("Number of QPD-samples")
    axes[1].set_xlabel("Number of QPD-samples")


fig, axes = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.3)


fig.text(0.5, 0.95, "(a) QAOA I", ha="center", fontweight="bold")
fig.text(0.5, 0.48, "(b) QAOA II", ha="center", fontweight="bold")


df = pd.read_json(data_path)
df1 = df[df["name"].isin(["qaoa1"])]
df2 = df[df["name"].isin(["qaoa2"])]


generate_figure(axes[0], df1.copy())
generate_figure(axes[1], df2.copy())
fig.savefig("benchmark/plots/sampling2.pdf", bbox_inches="tight")
