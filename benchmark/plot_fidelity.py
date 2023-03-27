import itertools
import matplotlib.pyplot as plt
import numpy as np

from benchstats import results_from_csv, get_results


def plot_fidelities(csv_files: dict[str, str], file_name: str):
    all_fidelities: dict[str, dict[int, list[float]]] = {}
    for exp_name, csv_file in csv_files.items():
        results = results_from_csv(csv_file)
        all_fidelities[exp_name] = get_results(results, "fidelity")
    
    experiments = sorted(all_fidelities.keys())

    all_qubits = [all_fidelities[exp].keys() for exp in experiments]
    nums_qubits = sorted(set(itertools.chain(*all_qubits)))

    means_per_experiment = {}
    std_per_experiment = {}
    for exp in experiments:
        new_means, new_std = [], []
        for num_qubits in nums_qubits:
            if num_qubits not in all_fidelities[exp]:
                new_means.append(-1.0)
                new_std.append(-1.0)
            else:
                new_means.append(float(np.mean(all_fidelities[exp][num_qubits])))
                new_std.append(float(np.std(all_fidelities[exp][num_qubits])))
        means_per_experiment[exp] = new_means
        std_per_experiment[exp] = new_std

    x = np.arange(len(nums_qubits))  # the label locations
    width = 1 / (len(experiments) + 1)  # the width of the bars
    multiplier = 1 - (len(experiments) - 1) * 0.5

    fig, ax = plt.subplots(layout="constrained")
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
    ax.set_xlabel("Number of qubits")
    ax.set_xticks(x + width, nums_qubits)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0.8, 1.0)
    # plt.savefig(file_name, dpi=300)
    plt.show()




plot_fidelities({"HamSim": "ham-sim.csv"}, "qft_fidelities.png")