from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .config import default_error_kw, get_style

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


def bar_comparison(
    ax: Axes,
    results: pd.DataFrame,
    keys: list[str],
    group_key: str,
    *,
    error: Literal["std", "sem", "ci95"] | None = "ci95",
) -> None:

    groups = results[group_key].dropna().unique()

    # x-axis = groups
    x = range(len(groups))

    total_width = 0.8
    num_keys = len(keys)
    bar_width = total_width / num_keys

    # offsets based on keys, not groups
    offsets = [(i - num_keys / 2) * bar_width + bar_width / 2 for i in range(num_keys)]

    keys = sorted(
        keys,
        key=lambda k: get_style(str(k)).sort_order,
    )

    for offset, key in zip(offsets, keys, strict=True):
        means = [results[results[group_key] == g][key].mean() for g in groups]

        errors = None
        match error:
            case "std":
                errors = [results[results[group_key] == g][key].std() for g in groups]
            case "sem":
                errors = [results[results[group_key] == g][key].sem() for g in groups]
            case "ci95":
                errors = [
                    (
                        1.96 * results[results[group_key] == g][key].sem()
                        if len(results[results[group_key] == g][key]) > 1
                        else 0
                    )
                    for g in groups
                ]

        config = get_style(str(key))

        ax.bar(
            [xi + offset for xi in x],
            means,
            width=bar_width,
            label=key,
            yerr=errors,
            hatch=config.hatch,
            facecolor=config.color,
            edgecolor="black",
            linewidth=1.5,
            error_kw=default_error_kw() if errors is not None else None,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
