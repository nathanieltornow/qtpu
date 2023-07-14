import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib import gridspec

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

def grouped_bar_plot(
    ax: plt.Axes,
    y: np.ndarray,
    yerr: np.ndarray,
    bar_labels: list[str],
    colors: list[str] | None = None,
    hatches: list[str] | None = None,
    spacing: float = 0.95,
    ):
    if colors is None:
        colors = sns.color_palette("pastel")
    if hatches is None:
        hatches = hatches = [
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

    assert len(y.shape) == len(yerr.shape) == 2
    assert y.shape == yerr.shape
    num_groups, num_bars = y.shape
    assert len(bar_labels) == num_bars

    bar_width = spacing / (num_bars + 1)
    x = np.arange(num_groups)

    for i in range(num_bars):
        y_bars = y[:, i]
        yerr_bars = yerr[:, i]

    color, hatch = colors[i % len(colors)], hatches[i % len(hatches)]
    
    ax.bar(
        x + (i * bar_width),
        y_bars,
        bar_width,
        hatch=hatch,
        label=bar_labels[i],
        yerr=yerr_bars,
        color=color,
        edgecolor="black",
        linewidth=1.5,
        error_kw=dict(lw=2, capsize=3),
    )
    ax.set_xticks(x + ((num_bars - 1) / 2) * bar_width)

def index_dataframe_mean_std(
    df: pd.DataFrame,
    xkey: str,
    xvalues: np.ndarray,
    ykey: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = (
        df.groupby(xkey)
        .agg({ykey: ["mean", "sem"]})
        .sort_values(by=[xkey])
        .reset_index()[[xkey, ykey]]
    )
    df = df.set_index(xkey)
    df = df.reindex(sorted(xvalues))
    df[ykey] = df[ykey].fillna(0.0)
    df = df.reset_index()
    return np.array(df[ykey]["mean"]), np.array(df[ykey]["sem"])

def data_frames_to_y_yerr(
    dataframes: list[pd.DataFrame],
    xkey: str,
    xvalues: np.ndarray,
    ykey: str,
    ykey_base: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if ykey_base is not None:
        for df in dataframes:
            df[ykey] = df[ykey] / df[ykey_base]
    
    mean_data, std_data = [], []
    for df in dataframes:
        mean, std = index_dataframe_mean_std(df, xkey, xvalues, ykey)
        mean_data.append(mean)
        std_data.append(std)

    return np.array(mean_data), np.array(std_data)


def calculate_figure_size(num_rows, num_cols):
    subplot_width_inches = 3.0  # Adjust this value based on your desired subplot width

    # Define the number of columns and rows of subplots
    num_cols = 2
    num_rows = 3

    # Calculate the total width and height based on the subplot width and number of columns and rows
    fig_width_inches = subplot_width_inches * num_cols
    fig_height_inches = fig_width_inches / 1.618 * num_rows  # Incorporate the golden ratio (1.618) for the height

    return fig_width_inches, fig_height_inches
