"""Decorator for logging function calls and results to a JSONL file."""

from __future__ import annotations

import datetime as dt
import functools
import inspect
import json
import platform
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TextIO, TypeVar

import git
import pandas as pd
import rich

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")


def log(file: TextIO | Path | str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function calls and their results.

    Args:
        file: Path or open file handle where log entries should be written.

    Returns:
        Decorator for logging function calls and results.
    """
    init_time = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Normalize to a Path if not already a file handle
    if isinstance(file, (str, Path)):
        log_path = Path(file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rich.print(
            f"[bold green]Logging to[/bold green] [cyan]{log_path.resolve()}[/cyan]"
        )
        file_handle: TextIO | None = None
    else:
        log_path = None
        file_handle = file

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            config = dict(bound.arguments)
            config.pop("self", None)
            config.pop("cls", None)

            result: R = fn(*args, **kwargs)

            entry: dict[str, Any] = {
                "config": config,
                "result": result,
                "id": str(uuid.uuid4())[:8],
                "func_name": fn.__name__,
                "init_time": init_time,
                "timestamp": dt.datetime.now(dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "host": platform.node(),
                "git_commit": _get_git_commit(),
            }

            line = json.dumps(entry) + "\n"

            if file_handle is not None:
                file_handle.write(line)
                file_handle.flush()
            else:
                assert log_path is not None
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(line)

            return result

        return wrapper

    return deco


def load_log(log_path: str | Path, *, normalize: bool = True) -> pd.DataFrame:
    """Load log entries from a JSONL log file.

    Args:
        log_path (str | Path): Path to the log file.
        normalize (bool, optional): Whether to flatten nested dicts into columns.

    Returns:
        pd.DataFrame: DataFrame of log entries.

    Raises:
        FileNotFoundError: If the log file does not exist.
    """
    log_path = Path(log_path)

    if not log_path.exists():
        msg = f"Log file {log_path} does not exist."
        raise FileNotFoundError(msg)

    log_df = pd.read_json(log_path, lines=True)

    if normalize:
        log_df = pd.json_normalize(log_df.to_dict(orient="records"))

    return log_df


def join_logs(
    log_paths: list[str | Path],
    *,
    how: str = "outer",
) -> pd.DataFrame:
    """Join multiple log files on their overlapping config columns.

    Args:
        log_paths (list[str | Path]): List of log file paths.
        how (str, optional): Merge method. Defaults to "outer".

    Returns:
        pd.DataFrame: Merged DataFrame of log entries.

    Raises:
        ValueError: If there are no overlapping config columns between logs.
    """
    # Load all logs as normalized DataFrames
    dfs = [load_log(p, normalize=True) for p in log_paths]

    if not dfs:
        return pd.DataFrame()

    # Start with the first df
    merged = dfs[0]

    # Iteratively merge with all subsequent dfs
    for df in dfs[1:]:

        # Find shared config.* columns
        config_cols = sorted(set(merged.columns).intersection(df.columns))
        config_cols = [c for c in config_cols if c.startswith("config.")]

        if not config_cols:
            msg = (
                f"No overlapping config columns between logs:\n"
                f"{merged.columns}\n---\n{df.columns}"
            )
            raise ValueError(msg)

        # Merge on overlapping config cols only
        merged = merged.merge(df, on=config_cols, how=how)
    return merged


def _get_git_commit() -> str:
    try:
        repo = git.Repo(search_parent_directories=True)
        return str(repo.head.object.hexsha[:7])
    except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
        return "unknown"


def _is_dirty() -> bool:
    try:
        repo = git.Repo(search_parent_directories=True)
        return bool(repo.is_dirty() or repo.untracked_files)
    except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
        return True
