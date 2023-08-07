from util import *

HIGHERISBETTER = "Higher is better ↑"
LOWERISBETTER = "Lower is better ↓"

from get_average import get_average


def print_stats(y: np.ndarray) -> None:
    print("median:", np.median(y))
    print("mean:", np.average(y))
    print("std:", np.std(y))
    print("min:", np.min(y))
    print("max:", np.max(y))


def plot_dep_min_stats() -> plt.Figure:
    DEP_MIN_DATA_3 = {
        "BV": "bench/results/greedy_dep_min/3/bv.csv",
        "VQE-1": "bench/results/greedy_dep_min/3/vqe_1.csv",
        "HS-2": "bench/results/greedy_dep_min/3/hamsim_2.csv",
        "TL-1": "bench/results/greedy_dep_min/3/twolocal_1.csv",
        "TL-2": "bench/results/greedy_dep_min/3/twolocal_2.csv",
        "TL-3": "bench/results/greedy_dep_min/3/twolocal_3.csv",
        "QAOA-B": "bench/results/greedy_dep_min/3/qaoa_b.csv",
        "QAOA-3": "bench/results/greedy_dep_min/3/qaoa_r3.csv",
        "QAOA-4": "bench/results/greedy_dep_min/3/qaoa_r4.csv",
    }

    dfs = [pd.read_csv(file) for file in DEP_MIN_DATA_3.values()]
    labels = list(DEP_MIN_DATA_3.keys())

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=WIDE_FIGSIZE, sharey=True)
    xvalues = [8, 16, 24]

    ax0.set_ylim(0, 1.2)

    y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array(xvalues), "num_deps", "num_deps_base"
    )
    grouped_bar_plot(ax0, y.T, yerr.T, labels, show_average_text=True)
    ax0.set_ylabel("Rel. Qubit Dependencies")
    ax0.set_title("(a) Qubit Dependencies", fontweight="bold", fontsize=FONTSIZE)
    ax0.set_xlabel("Number of Qubits")
    ax0.set_xticklabels(xvalues)
    _relative_plot(ax0)

    y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array(xvalues), "depth", "depth_base"
    )
    grouped_bar_plot(ax1, y.T, yerr.T, labels, show_average_text=True)
    ax1.set_ylabel("Rel. Circuit Depth")
    ax1.set_title("(b) Circuit Depth", fontweight="bold", fontsize=FONTSIZE)
    ax1.set_xlabel("Number of Qubits")
    ax1.set_xticklabels(xvalues)
    _relative_plot(ax1)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=10,
        frameon=False,
    )

    fig.text(
        0.51,
        0.9,
        LOWERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )

    return fig


def plot_dep_min_fidelity() -> plt.Figure:
    DEP_MIN_KOLKATA_REAL = {
        "HS-1": "bench/kolkata_results/dep_min/1/hamsim_1.csv",
        # "HS-2": "bench/kolkata_results/dep_min/1/hamsim_3.csv",
        "TL-1": "bench/kolkata_results/dep_min/1/twolocal_1.csv",
        "TL-2": "bench/kolkata_results/dep_min/1/twolocal_2.csv",
        "TL-3": "bench/kolkata_results/dep_min/1/twolocal_3.csv",
        "VQE-1": "bench/kolkata_results/dep_min/1/vqe_1.csv",
        "VQE-2": "bench/kolkata_results/dep_min/1/vqe_2.csv",
        "QAOA-2": "bench/kolkata_results/dep_min/1/qaoa_ba2.csv",
    }
    # DEP_MIN_KOLKATA_REAL = {
    #     "TL-1": "bench/kolkata_results/dep_min/2/twolocal_1.csv",
    #     "TL-2": "bench/kolkata_results/dep_min/2/twolocal_2.csv",
    #     "TL-3": "bench/kolkata_results/dep_min/2/twolocal_3.csv",
    #     "VQE-2": "bench/kolkata_results/dep_min/2/vqe_2.csv",
    # }

    xvalues = [6, 8, 10, 12]

    kolkata_dfs = [pd.read_csv(file) for file in DEP_MIN_KOLKATA_REAL.values()]

    # get_average(kolkata_dfs, "h_fid", "h_fid_base")

    titles = list(DEP_MIN_KOLKATA_REAL.keys())

    fig, axes = plt.subplots(1, len(titles), figsize=WIDE_FIGSIZE, sharey=True)
    fig.subplots_adjust(wspace=0.00)

    all_y = None

    for title, kolkata_df, ax in zip(titles, kolkata_dfs, axes):
        kolkata_base = kolkata_df.copy()
        kolkata_base["h_fid"] = kolkata_base["h_fid_base"]
        y, yerr = data_frames_to_y_yerr(
            [kolkata_df, kolkata_base],
            "num_qubits",
            np.array(xvalues),
            "h_fid",
        )
        if all_y is None:
            all_y = y[0] / y[1]
        else:
            all_y = np.concatenate((all_y, y[0] / y[1]))

        grouped_bar_plot(ax, y.T, yerr.T, ["QVM", "Baseline"])
        ax.set_title(title, fontweight="bold", fontsize=FONTSIZE)
        ax.set_xticklabels(xvalues)
        _relative_plot(ax)

    print_stats(all_y)

    axes[0].set_ylabel("Fidelity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.1, -0.08),
        ncol=100,
        frameon=False,
    )

    fig.text(
        0.51,
        0.98,
        HIGHERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )
    fig.text(0.5, -0.01, "Number of Qubits", ha="center")
    return fig


def plot_noisy_scale_stats() -> plt.Figure:
    NOISE_SCALE_KOLKATA = {
        # "GHZ": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/ghz.csv",
        "W-State": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/wstate.csv",
        "QSVM": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/qsvm.csv",
        "TL-1": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/twolocal_1.csv",
        "HS-1": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/hamsim_1.csv",
        "HS-2": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/hamsim_2.csv",
        "VQE-1": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/vqe_1.csv",
        "VQE-2": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/vqe_2.csv",
        "QAOA-B": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/qaoa_b.csv",
        "QAOA-2": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/qaoa_ba1.csv",
        # "VQE-3": "bench/results/noisy_scale_lol/ibmq_kolkata_vs_ibmq_kolkata/vqe_3.csv"
    }
    xvalues = [8, 16, 24]

    dfs = [pd.read_csv(file) for file in NOISE_SCALE_KOLKATA.values()]
    labels = list(NOISE_SCALE_KOLKATA.keys())

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=WIDE_FIGSIZE, sharey=True)

    y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array(xvalues), "num_cnots", "num_cnots_base"
    )
    grouped_bar_plot(
        ax0, y.T, yerr.T, labels, show_average_text=True, average_text_position=0.6
    )
    ax0.set_ylabel("Rel. Number of CNOTs")
    ax0.set_title("(a) CNOTs", fontweight="bold", fontsize=FONTSIZE)
    ax0.set_xlabel("Number of Qubits")
    ax0.set_xticklabels(xvalues)
    _relative_plot(ax0)

    y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array(xvalues), "depth", "depth_base"
    )
    grouped_bar_plot(
        ax1, y.T, yerr.T, labels, show_average_text=True, average_text_position=0.86
    )
    ax1.set_ylabel("Rel. Circuit Depth")
    ax1.set_title("(b) Circuit Depth", fontweight="bold", fontsize=FONTSIZE)
    ax1.set_xlabel("Number of Qubits")
    ax1.set_xticklabels(xvalues)
    _relative_plot(ax1)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=10,
        frameon=False,
    )
    fig.text(
        0.51,
        0.9,
        LOWERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )

    return fig


def plot_noisy_scale_fidelity() -> plt.Figure:
    NOISE_SCALE_PERTH_KOLKATA = {
        "HS-1": (
            "bench/perth_results/noisy_scale/hamsim_1.csv",
            "bench/kolkata_results/noisy_scale/hamsim_1.csv",
        ),
        "HS-2": (
            "bench/perth_results/noisy_scale/hamsim_2.csv",
            "bench/kolkata_results/noisy_scale/hamsim_2.csv",
        ),
        "TL-1": (
            "bench/perth_results/noisy_scale/twolocal_1.csv",
            "bench/kolkata_results/noisy_scale/twolocal_1.csv",
        ),
        "VQE-1": (
            "bench/perth_results/noisy_scale/vqe_1.csv",
            "bench/kolkata_results/noisy_scale/vqe_1.csv",
        ),
        "VQE-2": (
            "bench/perth_results/noisy_scale/vqe_2.csv",
            "bench/kolkata_results/noisy_scale/vqe_2.csv",
        ),
        "QSVM": (
            "bench/perth_results/noisy_scale/qsvm.csv",
            "bench/kolkata_results/noisy_scale/qsvm.csv",
        ),
        # "W-State": (
        #     "bench/perth_results/noisy_scale/wstate.csv",
        #     "bench/kolkata_results/noisy_scale/wstate.csv",
        # ),
        "QAOA-B": (
            "bench/perth_results/noisy_scale/qaoa_b.csv",
            "bench/kolkata_results/noisy_scale/qaoa_b.csv",
        ),
    }

    xvalues = [10, 14]

    kolkata_dfs = [pd.read_csv(file) for _, file in NOISE_SCALE_PERTH_KOLKATA.values()]
    perth_dfs = [pd.read_csv(file) for file, _ in NOISE_SCALE_PERTH_KOLKATA.values()]

    from get_average import get_average

    titles = list(NOISE_SCALE_PERTH_KOLKATA.keys())

    fig, axes = plt.subplots(1, len(titles), figsize=WIDE_FIGSIZE, sharey=True)
    fig.subplots_adjust(wspace=0.00)

    colors = sns.color_palette("pastel")
    colors = [colors[0], colors[2], colors[1]]

    all_y = None
    for title, kolkata_df, perth_df, ax in zip(titles, kolkata_dfs, perth_dfs, axes):
        kolkata_base = kolkata_df.copy()
        kolkata_base["h_fid"] = kolkata_base["h_fid_base"]
        y, yerr = data_frames_to_y_yerr(
            [perth_df, kolkata_df, kolkata_base],
            "num_qubits",
            np.array(xvalues),
            "h_fid",
        )

        if all_y is None:
            all_y = y[0] / y[2]
        else:
            all_y = np.concatenate((all_y, y[0] / y[2]))

        grouped_bar_plot(
            ax,
            y.T,
            yerr.T,
            [
                "QVM (IBM Perth, 7-qubit)",
                "QVM (IBM Kolkata, 27-qubit)",
                "Baseline (IBM Kolkata, 27-qubit)",
            ],
            colors=colors,
        )
        ax.set_title(title, fontweight="bold", fontsize=FONTSIZE)
        ax.set_xticklabels(xvalues)
        _relative_plot(ax)

    print_stats(all_y)

    axes[0].set_ylabel("Fidelity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=5,
        frameon=False,
    )
    fig.text(
        0.51,
        0.98,
        HIGHERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )
    fig.text(0.51, -0.01, "Number of Qubits", ha="center")
    return fig


def plot_dep_min_and_qubit_reuse() -> plt.Figure:
    DATA = {
        "TL-1": "bench/results/cutvreuse/ibm_perth/twolocal_1.csv",
        "HS-3": "bench/results/cutvreuse/ibm_perth/hamsim_3.csv",
        "VQE-3": "bench/results/cutvreuse/ibm_perth/vqe_3.csv",
    }

    dfs = [pd.read_csv(file) for file in DATA.values()]
    titles = list(DATA.keys())
    xvalues = [8, 10]

    fig, axes = plt.subplots(1, len(titles), figsize=COLUMN_FIGSIZE, sharey=True)
    fig.subplots_adjust(wspace=0.00)
    for title, df, ax in zip(titles, dfs, axes):
        df_base = df.copy()
        df_base["depth"] = df_base["depth_base"]
        y, yerr = data_frames_to_y_yerr(
            [df, df_base],
            "num_qubits",
            np.array(xvalues),
            "depth",
        )
        grouped_bar_plot(
            ax,
            y.T,
            yerr.T,
            [
                "CC pass",
                "DR & QR passes",
            ],
        )
        ax.set_title(title, fontweight="bold", fontsize=FONTSIZE)
        ax.set_xticklabels(xvalues)

    axes[0].set_ylabel("Circuit Depth")

    fig.text(0.51, -0.005, "Number of Qubits", ha="center")
    fig.text(
        0.51,
        0.98,
        LOWERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=10,
        frameon=False,
    )
    return fig


def plot_cut_vs_qubit_reuse() -> plt.Figure:
    DATA = {
        "W-State": "bench/results/reuse_vs_cut/ibm_perth/wstate.csv",
        "HS-2": "bench/results/reuse_vs_cut/ibm_perth/hamsim_2.csv",
        "VQE-2": "bench/results/reuse_vs_cut/ibm_perth/vqe_2.csv",
    }

    dfs = [pd.read_csv(file) for file in DATA.values()]
    titles = list(DATA.keys())
    xvalues = [8, 10]

    fig, axes = plt.subplots(1, len(titles), figsize=COLUMN_FIGSIZE, sharey=True)
    fig.subplots_adjust(wspace=0.00)
    for title, df, ax in zip(titles, dfs, axes):
        df_base = df.copy()
        df_base["depth"] = df_base["depth_base"]
        y, yerr = data_frames_to_y_yerr(
            [df, df_base],
            "num_qubits",
            np.array(xvalues),
            "depth",
        )
        grouped_bar_plot(
            ax,
            y.T,
            yerr.T,
            [
                "CC pass",
                "QR pass",
            ],
        )
        ax.set_title(title, fontweight="bold", fontsize=FONTSIZE)
        ax.set_xticklabels(xvalues)

    axes[0].set_ylabel("Circuit Depth")

    fig.text(
        0.51,
        0.98,
        LOWERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=10,
        frameon=False,
    )
    return fig


def plot_qr() -> plt.Figure:
    DATA1 = {
        "W-State": "bench/results/reuse_vs_cut/ibm_perth/wstate.csv",
        "HS-2": "bench/results/reuse_vs_cut/ibm_perth/hamsim_2.csv",
        "VQE-2": "bench/results/reuse_vs_cut/ibm_perth/vqe_2.csv",
    }

    dfs1 = [pd.read_csv(file) for file in DATA1.values()]
    titles1 = list(DATA1.keys())

    DATA2 = {
        "TL-1": "bench/results/cutvreuse/ibm_perth/twolocal_1.csv",
        "HS-3": "bench/results/cutvreuse/ibm_perth/hamsim_3.csv",
        "VQE-3": "bench/results/cutvreuse/ibm_perth/vqe_3.csv",
    }

    dfs2 = [pd.read_csv(file) for file in DATA2.values()]
    titles2 = list(DATA2.keys())

    xvalues = [8, 10]

    gridspec = dict(
        hspace=0.0, width_ratios=[1] * len(titles1) + [0.3] + [1] * len(titles2)
    )

    fig, axes = plt.subplots(
        1,
        len(titles1) + len(titles2) + 1,
        figsize=WIDE_FIGSIZE,
        sharey=True,
        gridspec_kw=gridspec,
    )
    axes[len(titles1)].axis("off")
    
    all_y = None

    for title, df, ax in zip(titles1, dfs1, axes[: len(titles1)]):
        df_base = df.copy()
        df_base["depth"] = df_base["depth_base"]
        y, yerr = data_frames_to_y_yerr(
            [df, df_base],
            "num_qubits",
            np.array(xvalues),
            "depth",
        )
        if all_y is None:
            all_y = y[1] / y[0]
        else:
            all_y = np.concatenate((all_y, y[1] / y[0]))

        grouped_bar_plot(
            ax,
            y.T,
            yerr.T,
            [
                "CC pass",
                "QR pass",
            ],
        )
        ax.grid(axis="y", linestyle="--")
        ax.set_title(title, fontweight="bold", fontsize=FONTSIZE)
        ax.set_xticklabels(xvalues)



    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.27, -0.18),
        ncol=10,
        frameon=False,
    )
    
    print_stats(all_y)
    
    all_y = None

    for title, df, ax in zip(titles2, dfs2, axes[len(titles1) + 1 :]):
        df_base = df.copy()
        df_base["depth"] = df_base["depth_base"]
        y, yerr = data_frames_to_y_yerr(
            [df, df_base],
            "num_qubits",
            np.array(xvalues),
            "depth",
        )
        if all_y is None:
            all_y = y[1] / y[0]
        else:
            all_y = np.concatenate((all_y, y[1] / y[0]))

        grouped_bar_plot(
            ax,
            y.T,
            yerr.T,
            [
                "CC pass",
                "DR + QR passes",
            ],
            colors=[sns.color_palette("pastel")[0], sns.color_palette("pastel")[3]],
            hatches=[
                "/",
                # "//",
                "\\\\",
            ],
        )
        ax.grid(axis="y", linestyle="--")
        ax.set_title(title, fontweight="bold", fontsize=FONTSIZE)
        ax.set_xticklabels(xvalues)

    print_stats(all_y)
    axes[0].set_ylabel("Circuit Depth")
    # axes[len(titles1)+1].set_ylabel("Circuit Depth")

    fig.text(0.27, -0.005, "Number of Qubits", ha="center")
    fig.text(
        0.51,
        0.99,
        LOWERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )

    handles, labels = axes[len(titles1) + 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.78,  -0.18),
        ncol=10,
        frameon=False,
    )
    
    fig.text(0.77, -0.005, "Number of Qubits", ha="center")

    fig.text(0.265, 0.98, "(a) CC vs. QR", ha="center", fontweight="bold", fontsize=FONTSIZE+2)

    fig.text(0.78, 0.98, "(b) CC vs. DR + QR", ha="center", fontweight="bold", fontsize=FONTSIZE+2)
    
    # for ax in axes[1:len(titles1)]:
    #     ax.yaxis.set_ticks([])
    #     ax.set_yticklabels([])
    #     ax.sharey(axes[0])
    # for ax in axes[len(titles1)+1:]:
    #     ax.sharey(axes[len(titles1)+1])

    return fig


def _relative_plot(ax: plt.Axes):
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
    ax.grid(axis="y", linestyle="--")
    ax.axhline(1, color="red", linestyle="-", linewidth=2)


def main():
    # fig = plot_dep_min_stats()

    # save_figure(fig, "dep_min_stats")

    # fig = plot_qr()
    # save_figure(fig, "qr")

    print("dep_min_fidelity")
    fig = plot_dep_min_fidelity()
    save_figure(fig, "dep_min_fidelity")

    # fig = plot_noisy_scale_stats()
    # save_figure(fig, "noisy_scale_stats")

    # print("noisy_scale_fidelity\n")
    # fig = plot_noisy_scale_fidelity()
    # save_figure(fig, "noisy_scale_fidelity")

    # fig = plot_dep_min_and_qubit_reuse()
    # save_figure(fig, "dep_min_and_qubit_reuse")

    # fig = plot_cut_vs_qubit_reuse()
    # save_figure(fig, "cut_vs_qubit_reuse")


if __name__ == "__main__":
    main()
