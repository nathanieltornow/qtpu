"""Decorator for iterating over a list of values for a specific parameter."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


P = ParamSpec("P")
R = TypeVar("R")


def foreach(**iters: Iterable[Any]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for iterating over a list of values for specific parameters.

    Args:
        **iters (Iterable[Any]): Keyword arguments where the key is the parameter
            name and the value is an iterable of values.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: The decorated function.
    """
    names = tuple(iters.keys())
    cols = tuple(iters.values())
    unset: Final = object()

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Caller fixed all names -> single call
            if all(n in kwargs for n in names):
                return fn(*args, **kwargs)

            last: object = unset
            for row in zip(*cols, strict=True):  # lengths must match
                call_kwargs = dict(kwargs)
                for n, v in zip(names, row, strict=True):
                    call_kwargs.setdefault(n, v)  # keep user-fixed values
                last = fn(*args, **call_kwargs)  # type: ignore[arg-type]

            if last is unset:
                msg = f"foreach(...): iterables produced no rows and not all params {names} were fixed via kwargs."
                raise ValueError(msg)
            return cast("R", last)

        return wrapper

    return deco


def retry(n: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to retry a function n times if it raises an exception.

    Args:
        n (int): Number of times to retry the function.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: The decorated function.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        msg = "n must be at least 1"
        raise ValueError(msg)

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            fn_ = fn  # local alias (micro-optimizes attribute lookups)
            for _ in range(n - 1):
                try:
                    return fn_(*args, **kwargs)
                except Exception:  # noqa: BLE001, PERF203, S110
                    pass
            # Final attempt without try/except so the real error propagates
            return fn_(*args, **kwargs)

        return wrapper

    return deco


def catch_failures(
    default: R,
    callback: Callable[[Exception], None] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to catch all exceptions and return a default value.

    Args:
        default (R): The default value to return if an exception is raised.
        callback (Callable[[Exception], None] | None): Optional callback to be called with the exception.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: The decorated function.
    """

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if callback is not None:
                    callback(e)
                return default

        return wrapper

    return deco
