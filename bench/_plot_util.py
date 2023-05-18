import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_single_experiment(
    ax,
    dataframe: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    key: str = "num_qubits",
):
    """Plots a comparison of columns for a single experiment."""
    assert len(columns) == len(labels)

    grouped_df = (
        dataframe.groupby([key])
        .agg([np.mean, np.std])
        .sort_values(by=[key])
        .reset_index()
    )

    x_values = np.arange(len(grouped_df))
    bar_width = 0.8 / len(columns)

    for i, column in enumerate(columns):
        bar_positions = [
            x + (i * bar_width) - ((len(columns) - 1) * bar_width) / 2 for x in x_values
        ]
        ax.bar(
            bar_positions,
            grouped_df[column]["mean"],
            width=bar_width,
            label=column,
            yerr=grouped_df[column]["std"],
            color=COLORS[i],
            ecolor="black",
            capsize=1,
            edgecolor="black",
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(grouped_df[key])


def plot_experiment_comparison(
    ax,
    dataframes: list[pd.DataFrame],
    column_name: str,
    labels: list[str],
    key: str = "num_qubits",
):
    """Plots a comparison of the given column_name for each dataframe in dataframes.

    Args:
        ax: The axis to plot on.
        dataframes (list[pd.DataFrame]): The dataframes to plot.
        column_name (str): The column to plot.
        labels (list[str]): The labels for each dataframe.
        key (str, optional): The key to aggregate for mean and std_err. Defaults to "num_qubits".
    """
    assert len(dataframes) == len(labels)

    for i, df in enumerate(dataframes):
        df_len = len(df)
        df.loc[:, "Experiment"] = np.repeat(labels[i], df_len)

    combined_df = pd.concat(dataframes)

    grouped_df = (
        combined_df.groupby(["Experiment", key]).agg([np.mean, np.std]).reset_index()
    )

    unique_experiments = grouped_df["Experiment"].unique()
    unique_keys = grouped_df[key].unique()
    unique_keys.sort()

    x_values = np.arange(len(unique_experiments))
    bar_width = 0.8 / len(unique_keys)

    for i, unique_key in enumerate(unique_keys):
        experiment_data = grouped_df[grouped_df[key] == unique_key]
        bar_positions = x_values + (i * bar_width)
        ax.bar(
            bar_positions,
            experiment_data[column_name]["mean"],
            yerr=experiment_data[column_name]["std"],
            width=bar_width,
            label=unique_key,
            color=COLORS[i],
            ecolor="black",
            capsize=1,
            edgecolor="black",
        )

    ax.set_xticks(x_values + (bar_width * (len(unique_keys) - 1)) / 2)
    ax.set_xticklabels(unique_experiments)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python plot_vqr.py <label1> <csv_file1> <label2> <csv_file2> ...")
        sys.exit(1)

    dataframes = [pd.read_csv(csv_file) for csv_file in sys.argv[2::2]]
    labels = sys.argv[1::2]
    for i, df in enumerate(dataframes):
        df["Experiment"] = labels[i]
