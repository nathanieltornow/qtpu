import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib import gridspec

FONTSIZE = 14
ISBETTER_FONTSIZE = FONTSIZE + 2
WIDE_FIGSIZE = (13, 2.8)
COLUMN_FIGSIZE = (6.5, 3.4)

plt.rcParams.update({"font.size": FONTSIZE})


def grouped_bar_plot(
    ax: plt.Axes,
    y: np.ndarray,
    yerr: np.ndarray,
    bar_labels: list[str],
    colors: list[str] | None = None,
    hatches: list[str] | None = None,
    show_average_text: bool = False,
    average_text_position: float = 1.05,
    spacing: float = 0.95,
    zorder: int = 2000,
):
    if colors is None:
        colors = sns.color_palette("pastel")
    if hatches is None:
        hatches = ["/", "\\", "//", "\\\\", "x", ".", ",", "*"]

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
            zorder=zorder,
        )
    ax.set_xticks(x + ((num_bars - 1) / 2) * bar_width)

    if show_average_text:
        for i, x_pos in enumerate(ax.get_xticks()):
            y_avg = np.average(y[i])
            text = f"{y_avg:.2f}"
            ax.text(x_pos, average_text_position, text, ha="center")


def bar_plot_dataframe(
    ax: plt.Axes,
    df: pd.DataFrame,
    bottom_df: pd.DataFrame | None = None,
    colors: list[str] | None = None,
    hatches: list[str] | None = None,
    spacing: float = 0.95,
    zorder: int = 2000,
):
    if colors is None:
        colors = sns.color_palette("pastel")
    if hatches is None:
        hatches = ["/", "\\", "//", "\\\\", "x", ".", ",", "*"]

    bar_labels = df.columns.values
    df = df.groupby(df.index).agg(["mean", "sem"]).sort_values(by=df.index.name)

    y = df.iloc[:, 0::2].values.T
    yerr = df.iloc[:, 1::2].values.T
    num_bars = y.shape[1]
    bar_width = spacing / (num_bars + 1)
    x = np.arange(num_bars)

    bottom = np.zeros(y.shape)

    if bottom_df is not None:
        # same index
        bottom_df = (
            bottom_df.groupby(bottom_df.index)
            .agg(["mean"])
            .sort_values(by=bottom_df.index.name)
        )
        assert bottom_df.index.equals(df.index)
        assert len(bottom_df.columns) == len(df.columns) / 2

        bottom = bottom_df.values.T

    for i in range(num_bars):
        y_bars = y[i]
        yerr_bars = yerr[i]

        color, hatch = colors[i % len(colors)], hatches[i % len(hatches)]

        ax.bar(
            x + (i * bar_width),
            y_bars,
            bar_width,
            hatch=hatch,
            bottom=bottom[i],
            label=bar_labels[i],
            yerr=yerr_bars,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            error_kw=dict(lw=2, capsize=3),
            zorder=zorder,
        )

    ax.set_xticks(x + ((num_bars - 1) / 2) * bar_width)


MARKER_STYLES = ["v", "o", "p", "^", "s", "D"]
LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
COLORS = sns.color_palette("pastel")


def line_plot(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    line_labels: list[str],
    colors: list[str] | None = None,
    markers: list[str] | None = None,
):
    if colors is None:
        colors = sns.color_palette("pastel")
    if markers is None:
        markers = ["v", "o", "p", "^", "s", "D"]

    assert len(y.shape) == len(yerr.shape) == 2
    assert y.shape == yerr.shape

    num_lines, _ = y.shape
    assert len(line_labels) == num_lines

    for i in range(num_lines):
        y_line = y[i]
        yerr_line = yerr[i]

        color, marker = colors[i % len(colors)], markers[i % len(markers)]

        ax.errorbar(
            x,
            y_line,
            yerr=yerr_line,
            label=line_labels[i],
            color=color,
            markeredgewidth=1.5,
            markeredgecolor="black",
            marker=marker,
            markersize=8,
            linewidth=2,
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            capsize=3,
        )

    ax.set_xticks(x)


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


def save_figure(fig: plt.Figure, exp_name: str):
    plt.tight_layout()
    fig.savefig(
        exp_name + ".pdf",
        bbox_inches="tight",
    )


# hatches = [
#             "/",
#             "\\",
#             "//",
#             "\\\\",
#             "x",
#             ".",
#             ",",
#             "*",
#             "o",
#             "O",
#             "+",
#             "X",
#             "s",
#             "S",
#             "d",
#             "D",
#             "^",
#             "v",
#             "<",
#             ">",
#             "p",
#             "P",
#             "$",
#             "#",
#             "%",
#         ]
