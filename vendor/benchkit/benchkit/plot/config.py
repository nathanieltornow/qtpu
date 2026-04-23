"""General plotting configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cycler import cycler


def colors() -> list[str]:
    """Return the default color cycle."""
    return [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
        "#000000",
    ]


def hatches() -> list[str]:
    """Return the default hatches."""
    return ["//", "\\\\", "---", "oo", "..", "xx", "++"]


def base_rc_params() -> dict[str, Any]:
    """Return matplotlib rc parameters for LaTeX-style plots."""
    font_size = 7
    return {
        "axes.prop_cycle": cycler("color", colors()),
        # --- LaTeX text ---
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "axes.titleweight": "bold",
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size + 1,
        # --- Embedding/fonts in vector backends ---
        "pdf.fonttype": 42,  # TrueType in PDF (note: ignored by usetex in many cases)
        "ps.fonttype": 42,
        # --- Lines (nicer look for paper figs) ---
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        # --- Axes / grid (helpful for barplots) ---
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        # --- Bars/Patches (crisp outlines that print well) ---
        "patch.edgecolor": "black",
        "patch.linewidth": 0.7,
        "patch.antialiased": True,
        # --- Hatching (pattern line thickness & color) ---
        "hatch.linewidth": 0.8,
        "hatch.color": "black",
        # --- Ticks (cleaner) ---
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # --- Savefig hygiene ---
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 400,
    }


GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


def default_error_kw() -> dict[str, Any]:
    """Return default error bar keyword arguments for matplotlib."""
    return {
        "elinewidth": 1.0,
        "capsize": 3,
        "capthick": 1.0,
        "zorder": 3,
        "ecolor": "black",
        "alpha": 0.9,
    }


def double_column_width() -> float:
    """Return the width of a double-column figure in inches."""
    return 7.0


def single_column_width() -> float:
    """Return the width of a single-column figure in inches."""
    return 3.5


def figsize(width: float, ratio: float = GOLDEN_RATIO) -> tuple[float, float]:
    """Return a figure size given width and height/width ratio.

    Args:
        width (float): Width of the figure in inches.
        ratio (float): Height/width ratio. Defaults to the golden ratio.

    Returns:
        tuple[float, float]: A tuple representing the width and height of the figure in inches
    """
    return (width, width / ratio)


@dataclass
class PlotStyle:
    """Style configuration for plots by label."""

    color: str | None = None
    alpha: float | None = None
    zorder: int = 2
    linestyle: str | None = None
    marker: str | None = None
    hatch: str | None = None
    sort_order: int = 100


class _StyleRegistry:
    def __init__(self) -> None:
        self._styles: dict[str, PlotStyle] = {}

    def register(self, label: str, style: PlotStyle) -> None:
        """Register a style for a given label.

        Args:
            label (str): The label to register the style for.
            style (Style): The style to register.

        Raises:
            ValueError: If the label is already registered.
        """
        if label in self._styles:
            msg = f"Style for label '{label}' is already registered."
            raise ValueError(msg)
        self._styles[label] = style

    def get(self, label: str) -> PlotStyle:
        """Get the style for a given label.

        Args:
            label (str): The label to get the style for.

        Returns:
            Style: The style for the given label.
        """
        if label not in self._styles:
            return PlotStyle()
        return self._styles[label]


_style_registry = _StyleRegistry()


register_style = _style_registry.register
get_style = _style_registry.get
