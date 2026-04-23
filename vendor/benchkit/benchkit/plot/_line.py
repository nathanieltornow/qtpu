from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .config import default_error_kw, get_style

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


def line_comparison(
    ax: Axes,
    results: pd.DataFrame,
    keys: list[str],
    group_key: str,
    *,
    error: Literal["std", "sem", "ci95"] | None = "ci95",
) -> None:
    """Plot a grouped line comparison.

    Args:
        ax: Matplotlib Axes.
        results: DataFrame with results.
        keys: Column names to plot as separate lines.
        group_key: Column used for x-axis grouping.
        error: Which error measure to display.
    """
    # X-axis groups
    groups = sorted(results[group_key].dropna().unique())

    # Sort keys by the plot style sort_order
    keys = sorted(keys, key=lambda k: get_style(str(k)).sort_order)

    # Numerical x-axis locations
    x = range(len(groups))

    # Precompute group slices for speed
    grouped = {g: results[results[group_key] == g] for g in groups}

    for key in keys:
        style = get_style(str(key))

        # Means per group
        means = [grouped[g][key].mean() for g in groups]

        # Error per group
        errors = None
        if error is not None:
            match error:
                case "std":
                    errors = [grouped[g][key].std() for g in groups]
                case "sem":
                    errors = [grouped[g][key].sem() for g in groups]
                case "ci95":
                    errors = [
                        (
                            1.96 * grouped[g][key].sem()
                            if len(grouped[g][key]) > 1
                            else 0
                        )
                        for g in groups
                    ]

        # Line itself
        ax.plot(
            x,
            means,
            label=key,
            color=style.color,
            alpha=style.alpha,
            zorder=style.zorder,
            linestyle=style.linestyle,
            marker=style.marker,
        )

        # Optional errorbars
        if errors is not None:
            ax.errorbar(
                x,
                means,
                yerr=errors,
                fmt="none",
                color=style.color,
                linewidth=1.0,
                capsize=3,
                zorder=style.zorder - 1,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
