import itertools
import matplotlib.pyplot as plt
import numpy as np

from benchstats import results_from_csv, get_results

# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)\

# change the font size
import matplotlib

matplotlib.rcParams.update({"font.size": 9})


# pastel colors
colors = ["darkorange", "royalblue", "forestgreen", "seagreen", "darkviolet", "darkred", "darkgoldenrod", "darkcyan", "darkmagenta"]


def plot_fidelities(
    ax, csv_files: list[tuple[str, str]], caption: str, lim_start: float = 0.0, range: tuple[int, int] = (2, 17)
):
    all_fidelities: dict[str, dict[int, list[float]]] = {}
    for exp_name, csv_file in csv_files:
        results = results_from_csv(csv_file)
        all_fidelities[exp_name] = get_results(results, "fidelity")

    experiments = list(files[0] for files in csv_files)

    all_qubits = [all_fidelities[exp].keys() for exp in experiments]
    nums_qubits = sorted(set(itertools.chain(*all_qubits)))
    nums_qubits = [n for n in nums_qubits if range[0] <= n <= range[1]]
    # nums_qubits = list(range(4, 17, 2))

    means_per_experiment = {}
    std_per_experiment = {}
    for exp in experiments:
        new_means, new_std = [], []
        for num_qubits in nums_qubits:
            if num_qubits not in all_fidelities[exp]:
                new_means.append(0.0)
                new_std.append(0.0)
            else:
                new_means.append(float(np.mean(all_fidelities[exp][num_qubits])))
                new_std.append(float(np.std(all_fidelities[exp][num_qubits])))
        means_per_experiment[exp] = new_means
        std_per_experiment[exp] = new_std

    x = np.arange(len(nums_qubits))  # the label locations
    width = 1 / (len(experiments) + 1)  # the width of the bars
    multiplier = 1 - (len(experiments) - 1) * 0.5

    for exp_name in experiments:
        offset = width * multiplier
        ax.bar(
            x + offset,
            means_per_experiment[exp_name],
            color=colors[experiments.index(exp_name)],
            yerr=std_per_experiment[exp_name],
            width=width,
            label=exp_name,
            ecolor="black",
            capsize=1,
            edgecolor="black",
        )
        for i, mean in enumerate(means_per_experiment[exp_name]):
            # check if the mean value is zero
            if mean == 0:
                # draw an x symbol instead of a bar
                ax.text(x[i] + offset, 0.022, 'x', ha='center', va='center', fontsize=7, color='red', fontweight='bold')
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # ax.set_ylabel("Hellinger Fidelity")
    ax.set_xticks(x + width, nums_qubits)
    # ax.set_xlabel(caption)
    # ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(lim_start, 1.0)
    ax.set_title(caption)
    # set legend to top right
    # ax.legend(loc="upper right", ncol=1)
    # plt.savefig(file_name, dpi=300)


fig = plt.figure(constrained_layout=True, figsize=(10, 2))
gs = fig.add_gridspec(1, 4)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)

plot_fidelities(
    ax1,
    [
        ("QVM (7-qubit QPUs)", "results/oslo_nairobi_scale/ghz.csv"),
        ("Guadalupe", "results/compare/ibmq_guadalupe/ghz.csv"),
        ("Oslo", "results/compare/ibmq_oslo/ghz.csv"),
        # ("27-qubit QPU (IMB Mumbai)", "results/ghz_mumbai.csv"),
    ],
    "(a) GHZ",
    0.0,
)
plot_fidelities(
    ax2,
    [
        ("QVM", "results/oslo_nairobi_scale/twolocal_1-rep.csv"),
        ("IBM Guadalupe", "results/compare/ibmq_guadalupe/twolocal_1.csv"),
        ("IBM Oslo", "results/compare/ibmq_oslo/twolocal_1.csv"),
    ],
    "(b) Two Local (1 rep)",
    0.0,
)
# plot_fidelities(
#     ax3,
#     [
#         ("QVM IBM Oslo", "results/oslo_nairobi_scale/qaoa_b.csv"),
#         ("IBM Guadalupe", "results/compare/ibmq_guadalupe/qaoa_b.csv"),
#         ("IBM Oslo", "results/compare/ibmq_oslo/qaoa_b.csv"),
#     ],
#     "(c) QAOA (Barbell Graph)",
#     0.0,
# )
plot_fidelities(
    ax3,
    [
        ("QVM", "results/oslo_nairobi_scale/twolocal_2-rep.csv"),
        ("Guadalupe", "results/compare/ibmq_guadalupe/twolocal-2.csv"),
        ("Oslo", "results/compare/ibmq_oslo/twolocal_2.csv"),
    ],
    "(c) Two Local (2 reps)",
    0.0,
    (2, 15)
)
plot_fidelities(
    ax4,
    [
        ("QVM", "results/oslo_nairobi_scale/qaoa_b.csv"),
        ("IBM Guadalupe", "results/compare/ibmq_guadalupe/qaoa_b.csv"),
        ("IBM Oslo", "results/compare/ibmq_oslo/qaoa_b.csv"),
    ],
    "(d) QAOA (Barbell Graph)",
    0.0,
)
# plot_fidelities(
#     ax4,
#     [
#         ("QVM IBM Oslo", "results/oslo_nairobi_scale/qaoa_l.csv"),
#         ("IBM Guadalupe", "results/compare/ibmq_guadalupe/qaoa_l.csv"),
#         ("IBM Oslo", "results/compare/ibmq_oslo/qaoa_l.csv"),
#     ],
#     "(d) QAOA (Ladder Graph)",
#     0.0,
# )

# plt.setp(ax1.get_xticklabels(), visible=False)

plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
# plot_fidelities(
#     ax3,
#     [
#         ("QVM with 7-qubit QPU (IMB Oslo)", "results/vqe_ibm_oslo.csv"),
#         ("27-qubit QPU (IMB Mumbai)", "results/vqe_mumbai.csv"),
#     ],
#     "(b) GHZ",
#     0.25,
# )
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower left", ncol=3, )
# fig.subplots_adjust(
#     top=1.0, bottom=0.0, left=0.2, right=0.985, hspace=0.4, wspace=0.08
# )


fig.supylabel("Hellinger Fidelity")
fig.supxlabel("Number of qubits")
plt.savefig("plot/scale_qpu.png", dpi=300)
