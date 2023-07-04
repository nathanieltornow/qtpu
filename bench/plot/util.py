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
    # Calculate the total number of axes in the plot
    num_axes = num_rows * num_cols

    # Determine the width and height of each axis
    axis_width = 3.5  # Adjust as needed
    axis_height = 2.6  # Adjust as needed

    # Determine the width and height of the figure based on the number of axes
    width = 1 * num_cols * axis_width
    height = 1 * num_rows * axis_height

    # # Adjust the figure size if there are more than 3 axes per row or 5 axes per column
    # if num_cols > 3:
    #     width += math.ceil((num_cols - 3) / 2) * axis_width
    # if num_rows > 5:
    #     height += math.ceil((num_rows - 5) / 2) * axis_height

    return width, height
