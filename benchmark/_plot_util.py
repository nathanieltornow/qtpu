import matplotlib.pyplot as plt


tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 10,
    # line width
    "lines.linewidth": 2,
    # set markersize
    "lines.markersize": 8,
    "lines.markeredgewidth": 1.5,
    "lines.markeredgecolor": "black",
    # bar edge width
}

# print(plt.rcParams.keys())

plt.rcParams.update(tex_fonts)


def postprocess_barplot(ax: plt.Axes) -> None:
    hatches = ["//", "\\\\", "||", "--", "++", "xx", "oo", "OO", "..", "**"]

    # for container in ax.containers:
