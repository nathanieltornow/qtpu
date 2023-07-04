import math

import pandas as pd

X_KEY = "num_qubits"

MARKER_STYLES = ["v", "p", "o", "^", "s", "D"]


def plot_lines(ax, keys: list[str], labels: list[str], dataframes: list[pd.DataFrame]):
    for ls, key in enumerate(keys):
        for df in dataframes:
            grouped_df = prepare_dataframe(df, key)
            x = grouped_df[X_KEY]
            y_mean = grouped_df[key]["mean"]
            y_error = grouped_df[key]["sem"]

            ax.errorbar(
                x,
                y_mean,
                # yerr=y_error,
                label=labels[ls],
                marker=MARKER_STYLES[ls],
                markersize=6,
                markeredgewidth=1.5,
                markeredgecolor="black",
                linestyle="-",
                linewidth=2,
                capsize=3,
                capthick=1.5,
                ecolor="black",
            )


def prepare_dataframe(df: pd.DataFrame, key: str) -> pd.DataFrame:
    res_df = df.loc[df[key] > 0.0]
    res_df = (
        df.groupby("num_qubits")
        .agg({key: ["mean", "sem"]})
        .sort_values(by=["num_qubits"])
        .reset_index()
    )
    return res_df


def calculate_figure_size(num_rows, num_cols):
    width = 4.5 * num_cols
    height = 3 * num_rows

    if num_cols > 3:
        width += math.ceil((num_cols - 3) / 2) * 1.5
    if num_rows > 5:
        height += math.ceil((num_rows - 5) / 2) * 1.5

    return width, height
