from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import get_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _scatter_comparison_discrete(
    ax: Axes,
    results: pd.DataFrame,
    x_key: str,
    y_key: str,
    group_key: str,
) -> None:
    groups = sorted(results[group_key].dropna().unique(), key=lambda g: get_style(str(g)).sort_order)

    for group in groups:
        group_data = results[results[group_key] == group]
        config = get_style(str(group))
        ax.scatter(
            group_data[x_key],
            group_data[y_key],
            label=group,
            color=config.color,
            marker=config.marker,
            zorder=config.zorder,
        )


def _scatter_comparison_continuous(
    ax: Axes,
    results: pd.DataFrame,
    x_key: str,
    y_key: str,
    group_key: str,
    cmap: str = "viridis",
) -> None:
    sub = results[[x_key, y_key, group_key]].dropna()
    config = get_style(group_key)
    sc = ax.scatter(
        sub[x_key],
        sub[y_key],
        c=sub[group_key],
        cmap=cmap,
        zorder=config.zorder,
        marker=config.marker,
    )
    plt.colorbar(sc, ax=ax, label=group_key)


def _get_linear_min_max(
    results: pd.DataFrame,
    x_key: str,
    y_key: str,
) -> tuple[float, float]:
    """Get min and max values for plotting a linear line through the origin.

    Returns (vmin, vmax) where vmin is the minimum of the x and y data,
        and vmax is the maximum of the y data.

    Args:
        results (pd.DataFrame): DataFrame containing the results data.
        x_key (str): Column name for x-axis values.
        y_key (str): Column name for y-axis values.

    Returns:
        tuple[float, float]: Minimum and maximum values for the axes.

    Raises:
        ValueError: If there are no valid data points for the given keys.
    """
    x_data = results[x_key].to_numpy()
    y_data = results[y_key].to_numpy()
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    if not np.any(mask):
        msg = f"No valid data points for keys '{x_key}' and '{y_key}'."
        raise ValueError(msg)

    x_min = np.min(x_data[mask])
    y_min, y_max = np.min(y_data[mask]), np.max(y_data[mask])
    return min(x_min, y_min), y_max


def scatter_comparison(
    ax: Axes,
    results: pd.DataFrame,
    x_key: str,
    y_key: str,
    group_key: str | None = None,
) -> None:
    """Create a scatter comparison plot on the given axes.

    Args:
        ax (Axes): Matplotlib Axes to plot on.
        results (pd.DataFrame): DataFrame containing the results data.
        x_key (str): Column name for x-axis values.
        y_key (str): Column name for y-axis values.
        group_key (str | None): Column name for grouping data points. If None, no grouping is applied.
    """
    vmin, vmax = _get_linear_min_max(results, x_key, y_key)

    if group_key is None:
        ax.scatter(results[x_key], results[y_key])
    elif pd.api.types.is_numeric_dtype(results[group_key]):
        _scatter_comparison_continuous(ax, results, x_key, y_key, group_key)
    else:
        _scatter_comparison_discrete(ax, results, x_key, y_key, group_key)

    ax.plot([vmin, vmax], [vmin, vmax], "k--", label="y=x")
