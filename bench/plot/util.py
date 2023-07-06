import math

import numpy as np
import pandas as pd

X_KEY = "num_qubits"

MARKER_STYLES = ["v", "o", "p", "^", "s", "D"]
LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def plot_lines(ax, keys: list[str], labels: list[str], dataframes: list[pd.DataFrame]):
    all_x = set()
    for ls, key in enumerate(keys):
        for df in dataframes:
            grouped_df = prepare_dataframe(df, key)
            x = grouped_df[X_KEY]
            all_x.update(set(x))
            y_mean = grouped_df[key]["mean"]
            y_error = grouped_df[key]["sem"]
            if np.isnan(y_error).any():
                y_error = None

            ax.errorbar(
                x,
                y_mean,
                yerr=y_error,
                label=labels[ls],
                # color=COLORS[ls],
                marker=MARKER_STYLES[ls],
                markersize=6,
                markeredgewidth=1.5,
                markeredgecolor="black",
                linestyle=LINE_STYLES[ls],
                linewidth=2,
                capsize=3,
                capthick=1.5,
                ecolor="black",
            )
    x = sorted(list(all_x))
    ax.set_xticks(x)

def prepare_dataframe(df: pd.DataFrame, key: str) -> pd.DataFrame:
    res_df = df.loc[df[key] > 0.0]
    res_df = (
        res_df.groupby("num_qubits")
        .agg({key: ["mean", "sem"]})
        .sort_values(by=["num_qubits"])
        .reset_index()
    )
    return res_df


def calculate_figure_size(num_rows, num_cols):
    subplot_width_inches = 3.0  # Adjust this value based on your desired subplot width

    # Define the number of columns and rows of subplots
    num_cols = 2
    num_rows = 3

    # Calculate the total width and height based on the subplot width and number of columns and rows
    fig_width_inches = subplot_width_inches * num_cols
    fig_height_inches = fig_width_inches / 1.618 * num_rows  # Incorporate the golden ratio (1.618) for the height

    return fig_width_inches, fig_height_inches
