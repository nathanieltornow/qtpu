import matplotlib.pyplot as plt

FONTSIZE = 12

tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": FONTSIZE,
    "font.size": FONTSIZE,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": FONTSIZE - 2,
    "xtick.labelsize": FONTSIZE - 2,
    "ytick.labelsize": FONTSIZE - 2,
    "axes.titlesize": 10,
    # line width
    "lines.linewidth": 2,
    # set markersize
    "lines.markersize": 6,
    "lines.markeredgewidth": 1.5,
    "lines.markeredgecolor": "black",
    # modify error bar settings
    "errorbar.capsize": 3,
    # bar edge width
}

# print(plt.rcParams.keys())

plt.rcParams.update(tex_fonts)


def postprocess_barplot(ax: plt.Axes, hatches: list[str] | None = None) -> None:
    if hatches is None:
        hatches = ["//", "\\\\", "oo", "xx", "oo", "OO"]

    color_to_hatch = {}

    i = 0
    for patch in ax.patches:
        if patch.get_facecolor() not in color_to_hatch:
            color_to_hatch[patch.get_facecolor()] = hatches[i % len(hatches)]
            i += 1

    for patch in ax.patches:
        patch.set_hatch(color_to_hatch[patch.get_facecolor()])
    # for container in ax.containers:


def postprocess_lineplot(ax: plt.Axes, linestyles=None, markers=None) -> None:
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    if markers is None:
        markers = ["o", "s", "^", "D", "v", "p"]

    color_to_style = {}
    color_to_marker = {}
    i = 0

    for line in ax.get_lines():
        color = line.get_color()
        if color not in color_to_style:
            color_to_style[color] = linestyles[i % len(linestyles)]
            color_to_marker[color] = markers[i % len(markers)]
            i += 1

        line.set_linestyle(color_to_style[color])
        line.set_marker(color_to_marker[color])
        line.set_markersize(6)
        line.set_markeredgewidth(1)
        line.set_markeredgecolor("black")
        line.set_markerfacecolor(color)
