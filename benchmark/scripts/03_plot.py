import pandas as pd
from benchmark._plot_util import *
from matplotlib.ticker import FixedLocator


data_path = "benchmark/results/end_to_end_estimator.json"
FIG_SIZE = (4, 2.1)


def plot_end2end(ax):
    df = pd.read_json("benchmark/results/end_to_end_1.json")
    df = df[df["name"] != "qaoa1"]
    # df["num_qubits"] = df["num_qubits"] // 10 * 10
    # df1 = df[(df["name"].str.startswith("qaoa")) & (df["m"] == 1)]
    # df2 = df[df["name"].isin(["vqe", "qml"]) & (df["reps"] == 1)]
    # df2 = df[  & (df["name"].isin(["qaoa1", "qaoa2"]))]
    # df = pd.concat([df1, df2])
    df.drop(columns=["name"], inplace=True)

    df["qtpu"] = df["qtpu_gpu_run"] + df["qtpu_gpu_post"]
    df["ckt"] = df["ckt_run"] + df["ckt_post"]
    df["cutensor"] = df["cutensor_exec"] + df["cutensor_compile"]

    df_mean = df.groupby(["num_qubits"]).mean()
    df_std = df.groupby(["num_qubits"]).std()

    markers = ["o", "s", "v"]

    for y, marker in zip(["qtpu", "ckt", "cutensor"], markers):
        df_mean.plot.line(
            y=[y],
            yerr=df_std,
            ax=ax,
            linewidth=2,
            capsize=3,
            marker=marker,
            linestyle="-",
            color="black" if y == "cutensor" else None,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Runtime [s]")
    ax.set_xlabel("Number of qubits")
    ax.legend(["QTPU", "QAC", "CuTensorNet"])


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
    # ax.set_title("(b) End-to-end runtime", fontweight="bold", color="black")
    # ax.legend(["VQE", "QML", "QAOA I", "QAOA II", "CuTensorNet"])


def generate_figure():
    fig, ax1 = plt.subplots(1, 1, figsize=(4.5, 2.1))
    # plot_end2end(ax1)
    plot_end2end_2(ax1)
    # handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        ["VQE", "QML", "QAOA I", "QAOA II", "CuTensorNet"],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.22),
    )

    # plot_end2end(ax1)

    fig.savefig("benchmark/plots/03_end_to_end.pdf", bbox_inches="tight")


# def print_summary():
#     df = pd.read_json(data_path)

#     print(df.aggregate(["mean", "std", "min", "max"]).to_json(indent=4))


generate_figure()
# print_summary()
