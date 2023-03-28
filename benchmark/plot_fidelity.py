import itertools
import matplotlib.pyplot as plt
import numpy as np

from benchstats import results_from_csv, get_results


def plot_fidelities(ax, csv_files: list[tuple[str, str]], caption: str, lim_start: float = 0.0):
    all_fidelities: dict[str, dict[int, list[float]]] = {}
    for exp_name, csv_file in csv_files:
        results = results_from_csv(csv_file)
        all_fidelities[exp_name] = get_results(results, "fidelity")

    experiments = list(files[0] for files in csv_files)

    all_qubits = [all_fidelities[exp].keys() for exp in experiments]
    nums_qubits = sorted(set(itertools.chain(*all_qubits)))

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
        rects = ax.bar(
            x + offset,
            means_per_experiment[exp_name],
            yerr=std_per_experiment[exp_name],
            width=width,
            label=exp_name,
        )
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Hellinger Fidelity")

    ax.set_xticks(x + width, nums_qubits)
    ax.set_xlabel(caption)
    # ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(lim_start, 1.0)
    # set legend to top right
    # ax.legend(loc="upper right", ncol=1)
    # plt.savefig(file_name, dpi=300)


fig, (ax1, ax2) = plt.subplots(2)

plot_fidelities(
    ax1,
    [
        ("QVM with 7-qubit QPU (IMB Oslo)", "results/ham_ibm_oslo.csv"),
        ("27-qubit QPU (IMB Mumbai)", "results/ham_mumbai.csv"),
    ],
    "(a) Hamiltonian Simulation",
    0.9,
)
plot_fidelities(
    ax2,
    [
        ("QVM with 7-qubit QPU (IMB Oslo)", "results/ghz_ibm_oslo.csv"),
        ("27-qubit QPU (IMB Mumbai)", "results/ghz_mumbai.csv"),
    ],
    "(b) GHZ",
    0.0,
)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center")
plt.xlabel("Number of qubits")
plt.savefig("scale_qpu.png", dpi=300)
