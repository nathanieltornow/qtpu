"""Utils for plotting usual plots."""

from __future__ import annotations

from ._bar import bar_comparison
from ._line import line_comparison
from ._scatter import scatter_comparison
from .pplot import pplot

__all__ = ["bar_comparison", "line_comparison", "pplot", "scatter_comparison"]
