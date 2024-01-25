import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def postprocess_barplot(ax: plt.Axes) -> None:
    hatches = ["/", "\\", "//", "\\\\", "x", ".", ",", "*"]
    num_xticks = len(ax.get_xticks())
    num_bars = len(ax.get_legend_handles_labels()[0])
    patch_idx_to_hatch_idx = np.arange(num_bars).repeat(num_xticks)
    for i, patch in enumerate(ax.patches):
        patch.set_hatch(hatches[patch_idx_to_hatch_idx[i] % len(hatches)])
