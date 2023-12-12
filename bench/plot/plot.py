from bench.plot.util import *
import os
from bench.plot.bar_plot import *
from bench.plot.data import *

HIGHERISBETTER = "Higher is better ↑"
LOWERISBETTER = "Lower is better ↓"

from bench.plot.get_average import get_average


# sns.set_theme(style="whitegrid", color_codes=True)
colors = sns.color_palette("pastel")

plt.rcParams.update({"font.size": 12})


def insert_column(df):
    df["total_runtime"] = df["run_time"] + df["knit_time"]

    return df


def dataframe_out_of_columns(dfs, lines, columns):
    merged_df = pd.DataFrame()

    merged_df["num_qubits"] = dfs[0]["num_qubits"].copy()
    merged_df.set_index("num_qubits")

    for i, f in enumerate(dfs):
        merged_df[lines[i]] = f[columns].copy()

    # merged_df.reset_index(drop = True, inplace = True)
    merged_df.set_index("num_qubits", inplace=True)

    return merged_df


def plot_endtoend_runtimes():
    dfs = [pd.read_csv(file) for file in SCALE_SIM_TIME.values()]
    dfs_mem = [pd.read_csv(file) for file in SCALE_SIM_MEMORY.values()]
    dfs_knit_time = [pd.read_csv(file) for file in SCALE_SIM_KNIT_TIME.values()]

    pivot_df = dfs_knit_time[0].pivot_table(
        index="num_threads", columns="num_vgates", values="time", aggfunc="mean"
    )
    pivot_df.columns = ["1vg", "2vg", "3vg", "4vg"]
    pivot_df = pivot_df.rename_axis("num_qubits")
    print(pivot_df)
    # print(pivot_df.keys())

    lines = [s.split("-")[-1] for s in SCALE_SIM_TIME.keys()]

    titles = [
        "(a) Εnd-to-end Runtime",
        "(b) Runtime Breakdown",
        "(c) Knitter's Scalability",
        "(d) Memory Consumption",
    ]

    dfs = [insert_column(i) for i in dfs]
    big_dfs = dataframe_out_of_columns(dfs, lines, ["total_runtime"])

    dfs_mem_new = pd.DataFrame()
    dfs_mem_new["num_qubits"] = dfs_mem[0]["num_qubits"].copy()
    dfs_mem_new["Baseline"] = dfs_mem[0]["h_fid"]
    dfs_mem_new["QVM"] = dfs_mem[0]["h_fid_base"]
    dfs_mem_new["CutQC"] = dfs_mem[0]["tv_fid"]
    dfs_mem_new.set_index("num_qubits", inplace=True)
    # print(dfs_mem_new)

    dfs_ratio = pd.DataFrame()
    dfs_ratio["qpu_size"] = [15, 20, 25]
    dfs_ratio.set_index("qpu_size")

    dfs_ratio["simulation"] = [d.loc[4].at["run_time"] for d in dfs]
    dfs_ratio["knitting"] = [d.loc[4].at["knit_time"] for d in dfs]

    keys = dfs_ratio.keys()
    keys = keys[1:]

    custom_plot_dataframes(
        dataframes=[big_dfs, dfs_ratio, pivot_df, dfs_mem_new],
        keys=[big_dfs.keys(), keys, pivot_df.keys(), dfs_mem_new.keys()],
        labels=[
            big_dfs.keys(),
            dfs_ratio["qpu_size"].tolist(),
            pivot_df.keys(),
            dfs_mem_new.keys(),
        ],
        titles=titles,
        ylabel=["Runtime [s]", "Runtime [s]", "Knitting Time [s]", "Memory [GBs]"],
        xlabel=[
            "Number of Qubits",
            "QPU Size [Number of Qubits]",
            "Number of Threads",
            "Number of Qubits",
        ],
        output_file="./end_to_end.pdf",
        logscale=True,
        nrows=1,
    )


hatches = [
    "/",
    "\\",
    "//",
    "\\\\",
    "x",
    ".",
    ",",
    "*",
    "o",
    "O",
    "+",
    "X",
    "s",
    "S",
    "d",
    "D",
    "^",
    "v",
    "<",
    ">",
    "p",
    "P",
    "$",
    "#",
    "%",
]


def custom_plot_dataframes(
    dataframes: list[pd.DataFrame],
    keys: list[list[str]],
    labels: list[list[str]],
    titles: list[str],
    ylabel: list[str],
    xlabel: list[str],
    output_file: str = "noisy_scale.pdf",
    nrows: int = 2,
    logscale=False,
) -> None:
    ncols = len(dataframes)
    fig = plt.figure(figsize=(13, 3.4))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

    axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

    axis[0].set_yscale("log")
    axis[1].set_yscale("log")
    axis[2].set_yscale("log")
    axis[2].set_xscale("log")
    axis[3].set_yscale("log")

    # axis[1].sharey(axis[0])
    # axis[2].set_xlim([10, 30])
    axis[1].set_ylim([0, 50000])
    # axis[2].set_yticks([0, 1, 10, 100, 1000], [0, 1, 10, 100, 1000])
    axis[2].set_ylim([3 * 10 ** (-2), 10**4])
    # axis[2].set_yscale("log")

    for i, ax in enumerate(axis):
        ax.set_ylabel(ylabel=ylabel[i])
        ax.set_xlabel(xlabel=xlabel[i])

    # print(keys)
    plot_lines(axis[0], keys[0], labels[0], [dataframes[0]])
    axis[0].legend()
    axis[0].set_title(titles[0], fontsize=12, fontweight="bold")

    print(keys[2])
    plot_lines(axis[2], keys[2], labels[2], [dataframes[2]])
    axis[2].legend(ncols=2)
    axis[2].set_title(titles[2], fontsize=12, fontweight="bold")

    plot_lines(axis[3], keys[3], labels[3], [dataframes[3]])
    axis[3].legend()
    axis[3].set_title(titles[3], fontsize=12, fontweight="bold")

    num_vgates = dataframes[1]["qpu_size"].tolist()
    simulation = dataframes[1]["simulation"].tolist()
    knitting = dataframes[1]["knitting"].tolist()
    data = {
        "Simulation": simulation,
        "Knitting": knitting,
    }

    x = np.array([15, 20, 25])
    # x = np.arange(len(num_vgates))  # the label locations
    # width = 0.25  # the width of the bars
    # multiplier = 0
    y = np.array(
        [
            [9.52130384114571, 120.0079321230296, 801.0942367650568],
            [11.77336971112527, 726.3718322570203, 208.40429024997866],
            [1.7376548638567328, 5857.7779290829785, 305.2052580610034],
        ]
    )

    yerr = np.array(
        [
            [1.3718322570203, 6.270605635945685, 41.68920839508064],
            [2.7376548638567328, 33.503638901049, 8.03563788096653],
            [0.2052580610034, 155.2813523421064, 22.93891781999264],
        ]
    )

    axis[1].set_xticklabels(x)
    grouped_bar_plot(axis[1], y, yerr, ["Compilation", "Simulation", "Knitting"])
    # axis[1].set(ylabel=None)
    axis[1].legend(loc="upper left", ncols=1)


    # axis[1].set_yticks(np.logspace(1, 5, base=10, num=5, dtype="int"))
    axis[1].set_title(titles[1], fontsize=12, fontweight="bold")

    
    axis[0].grid(axis="y", linestyle="--", zorder=-1)
    axis[1].grid(axis="y", linestyle="--", zorder=-1)
    axis[2].grid(axis="y", linestyle="--", zorder=-1)
    axis[3].grid(axis="y", linestyle="--", zorder=-1)

    fig.text(
        0.51,
        .98,
        LOWERISBETTER,
        ha="center",
        fontweight="heavy",
        color="midnightblue",
        fontsize=ISBETTER_FONTSIZE,
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")




def main():
    plot_endtoend_runtimes()


if __name__ == "__main__":
    main()
