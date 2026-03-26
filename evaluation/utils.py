"""Evaluation utilities for qTPU benchmarks.

Provides lightweight logging and plotting helpers so the evaluation
scripts are self-contained and don't depend on external benchmarking
frameworks.
"""

from __future__ import annotations

import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------


def log_result(path: str | Path, config: dict[str, Any], result: dict[str, Any] | None) -> None:
    """Append one ``{"config": …, "result": …}`` line to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "result": result,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load_results(path: str | Path) -> pd.DataFrame:
    """Load a JSONL log into a DataFrame with ``config.*`` / ``result.*`` columns."""
    if not os.path.exists(path):
        return pd.DataFrame()
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows, sep=".")


# ---------------------------------------------------------------------------
# Timeout helper (Unix only – fine for research workloads)
# ---------------------------------------------------------------------------


def run_with_timeout(fn: Callable[[], Any], timeout_secs: int, default: Any = None) -> Any:
    """Run *fn()* with a wall-clock timeout.  Returns *default* on timeout."""
    if not hasattr(signal, "SIGALRM"):
        return fn()  # Windows: no timeout support

    def _handler(_signum: int, _frame: Any) -> None:
        msg = f"Timed out after {timeout_secs}s"
        raise TimeoutError(msg)

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_secs)
    try:
        return fn()
    except TimeoutError:
        return default
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Paper color palette — matched to the OSDI figures.
COLORS = [
    "#4472C4",  # blue  (qTPU)
    "#ED7D31",  # orange
    "#A5A5A5",  # gray
    "#C44E52",  # red/pink (Mitiq / baselines)
    "#9467BD",  # purple
    "#D4A373",  # tan (Batch)
    "#E377C2",  # pink
    "#7F7F7F",  # dark gray
    "#BCBD22",  # olive
    "#17BECF",  # cyan
]


def colors() -> list[str]:
    """Return the default color palette."""
    return list(COLORS)


def single_column_width() -> float:
    """Width in inches for a single-column figure (ACM/USENIX style)."""
    return 3.5


def double_column_width() -> float:
    """Width in inches for a double-column figure."""
    return 7.0


class PlotStyle:
    """Lightweight style descriptor for bar/line plots."""

    def __init__(self, color: str = "#4472C4", hatch: str = "") -> None:
        self.color = color
        self.hatch = hatch


_STYLES: dict[str, PlotStyle] = {}


def register_style(name: str, style: PlotStyle) -> None:
    """Register a named plot style."""
    _STYLES[name] = style


def get_style(name: str) -> PlotStyle:
    """Retrieve a registered style (returns a default if not found)."""
    return _STYLES.get(name, PlotStyle())


def setup_paper_style() -> None:
    """Configure matplotlib rcParams to match the OSDI paper figures."""
    import matplotlib as mpl

    mpl.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        # Lines & patches
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "patch.linewidth": 0.5,
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        # Layout
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3,
    })
