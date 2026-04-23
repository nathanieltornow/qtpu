"""SQLite-backed caching decorator."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import pickle  # noqa: S403
import sqlite3
import time
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast, overload

from benchkit.config import data_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


@overload
def cache(
    func: Callable[P, R],
    /,
) -> Callable[P, R]: ...
@overload
def cache(
    *,
    name: str | None = None,
    ttl: float | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def cache(
    func: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    ttl: float | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """SQLite-backed caching decorator with optional TTL.

    Args:
        func: The function to be decorated (internal, allows bare @cache usage).
        name: Optional name for the cache file. Defaults to the function name.
        ttl: Time-to-live in seconds. If None, entries never expire.

    Returns:
        A decorated function or a decorator.
    """

    def decorator(inner_func: Callable[P, R]) -> Callable[P, R]:
        cache_name = name or inner_func.__name__
        cache_dir: Path = data_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dbfile: Path = cache_dir / f"{cache_name}.db"

        conn = sqlite3.connect(dbfile)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                   key TEXT PRIMARY KEY,
                   value BLOB,
                   created_at REAL NOT NULL
               )"""
        )

        sig = inspect.signature(inner_func)

        @functools.wraps(inner_func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # canonicalize inputs
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs: dict[str, Any] = dict(bound.arguments)
            inputs.pop("self", None)
            inputs.pop("cls", None)

            raw = json.dumps(inputs, sort_keys=True, default=str).encode()
            key = hashlib.sha256(raw).hexdigest()

            now = time.time()
            cur = conn.execute("SELECT value, created_at FROM cache WHERE key=?", (key,))
            row = cur.fetchone()
            if row is not None:
                value, created_at = row
                if ttl is None or (now - created_at) < ttl:
                    logger.debug("Cache hit for %s with inputs %s", cache_name, inputs)
                    return cast("R", pickle.loads(value))  # noqa: S301
                logger.debug("Cache expired for %s with inputs %s", cache_name, inputs)

            # cache miss or expired
            logger.debug("Cache miss for %s with inputs %s", cache_name, inputs)
            result: R = inner_func(*args, **kwargs)

            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, pickle.dumps(result), now),
            )
            conn.commit()
            return result

        return wrapper

    # bare decorator: @cache
    if func is not None:
        return decorator(func)
    # decorator factory: @cache(...)
    return decorator
