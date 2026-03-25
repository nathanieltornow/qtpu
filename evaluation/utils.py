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
# Plotting helpers (replace benchkit.plot.config)
# ---------------------------------------------------------------------------

# Matplotlib tab10 palette – matches the default used in most frameworks.
COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
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

    def __init__(self, color: str = "#1f77b4", hatch: str = "") -> None:
        self.color = color
        self.hatch = hatch


_STYLES: dict[str, PlotStyle] = {}


def register_style(name: str, style: PlotStyle) -> None:
    """Register a named plot style."""
    _STYLES[name] = style


def get_style(name: str) -> PlotStyle:
    """Retrieve a registered style (returns a default if not found)."""
    return _STYLES.get(name, PlotStyle())
