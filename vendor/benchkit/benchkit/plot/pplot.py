"""Utility functions for pretty plots."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

import matplotlib.pyplot as plt
import rich
from matplotlib.figure import Figure

from .config import base_rc_params

if TYPE_CHECKING:
    from collections.abc import Callable


R = TypeVar("R", bound=Figure | Iterable[Figure])
P = ParamSpec("P")


@overload
def pplot(
    _fn: Callable[P, R],
) -> Callable[P, R]: ...


@overload
def pplot(
    _fn: None = None,
    *,
    dir_path: Path | str = "plots",
    plot_name: str | None = None,
    custom_rc: dict[str, Any] | None = None,
    extensions: list[str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def pplot(
    _fn: Callable[P, R] | None = None,
    *,
    dir_path: Path | str = "plots",
    plot_name: str | None = None,
    custom_rc: dict[str, Any] | None = None,
    extensions: list[str] | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to save pretty plots.

    Args:
        dir_path (Path | str): Directory to save plots. Defaults to "plots".
        plot_name (str | None): Name of the plot file. If None, uses the function name.
        custom_rc (dict[str, Any] | None): Custom matplotlib rc parameters.
        extensions (list[str] | None): List of file extensions to save the plots.

    Returns:
        Callable: Decorator function that wraps the plotting function.
    """
    if extensions is None:
        extensions = ["pdf"]

    custom_rc = custom_rc or {}
    rc_params = base_rc_params()
    rc_params.update(custom_rc)

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with plt.rc_context(rc=rc_params):
                result = fn(*args, **kwargs)
                for extension in extensions:
                    _save_figures(
                        result,
                        dir_path=dir_path,
                        fname=plot_name or fn.__name__,
                        extension=extension,
                    )
                return result

        return wrapper

    if callable(_fn):
        return decorator(_fn)
    return decorator


def _save_figures(
    figs: Figure | Iterable[Figure],
    dir_path: Path | str,
    fname: str,
    extension: str = "pdf",
) -> None:
    """Save figure(s) to PDF in a dated directory structure."""
    date_str = datetime.now().astimezone().strftime("%Y-%m-%d-%H-%M")
    out_dir = Path(dir_path) / fname / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_one(fig: object, filename: str) -> None:
        if not isinstance(fig, Figure):
            return
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=400, bbox_inches="tight")
        rich.print(f":floppy_disk: Saved plot to [bold]{out_dir / filename}[/bold]")

    if isinstance(figs, Figure):
        _save_one(figs, f"{fname}.{extension}")
    elif isinstance(figs, Iterable):
        for i, maybe_fig in enumerate(figs):
            _save_one(maybe_fig, f"{fname}_{i}.{extension}")
