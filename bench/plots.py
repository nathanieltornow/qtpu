import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from _plot_util import plot_experiment_comparison


def _get_data_frames_from_directory(path: str) -> tuple[list[str], list[pd.DataFrame]]:
    files_in_dir = os.listdir(path)
    csv_files = sorted(f for f in files_in_dir if f.endswith(".csv"))
    data_frames = [pd.read_csv(f"{path}/{f}") for f in csv_files]
    labels = [f.split(".")[0] for f in csv_files]
    return labels, data_frames


def plot_relative_fidelity(
    data_dir: str, output_path: str, nums_qubits: list[int] = [6, 10, 14]
):
    labels, dataframes = _get_data_frames_from_directory(data_dir)

    for df in dataframes:
        df["relative_fidelity"] = df["base_fidelity"] / df["vroute_fidelity"]
        # filter the dataframe to only include the desired nums_qubits

    filtered_dataframes = []
    for df in dataframes:
        condition = df["num_qubits"].isin(nums_qubits)
        df = df.loc[condition]
        filtered_dataframes.append(df)
    dataframes = filtered_dataframes

    fig, ax = plt.subplots()

    plot_experiment_comparison(
        ax,
        dataframes,
        "relative_fidelity",
        labels,
    )

    # plot a dashed line at 1.0
    ax.axhline(y=1.0, color="darkred", linestyle="-")
    plt.ylim(0.9)

    ax.legend(title="n_qubits")
    plt.ylabel("Relative Fidelity")
    plt.savefig(output_path)


def plot_cx_overhead(
    data_dir: str, output_path: str, nums_qubits: list[int] = [6, 10, 14]
):
    labels, dataframes = _get_data_frames_from_directory(data_dir)

    for df in dataframes:
        df["cx_overhead"] = df["base_num_cx"] / df["vroute_num_cx"]
        # filter the dataframe to only include the desired nums_qubits

    filtered_dataframes = []
    for df in dataframes:
        condition = df["num_qubits"].isin(nums_qubits)
        df = df.loc[condition]
        filtered_dataframes.append(df)
    dataframes = filtered_dataframes

    fig, ax = plt.subplots()

    plot_experiment_comparison(
        ax,
        dataframes,
        "cx_overhead",
        labels,
    )
    # plot a dashed line at 1.0
    ax.axhline(y=1.0, color="darkred", linestyle="-")

    ax.legend(title="n_qubits")
    plt.ylabel("CNOT Overhead")
    plt.ylim(0.9)
    plt.savefig(output_path)


def main():
    if len(sys.argv) < 4:
        print("usage: python plots.py <plot_name> <data_directory> <output_path>")
        sys.exit(1)

    if sys.argv[1] == "cx":
        plot_cx_overhead(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "fidelity":
        plot_relative_fidelity(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
