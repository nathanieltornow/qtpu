"""Timeout decorator using persistent, lazily-started worker subprocesses."""

from __future__ import annotations

import functools
import multiprocessing as mp
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import cloudpickle

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection

P = ParamSpec("P")
R = TypeVar("R")


# ===============================================================
# Worker process entrypoint
# ===============================================================
def _worker_loop(conn: Connection) -> None:
    """Entry point for worker subprocess.

    Repeatedly receives ``(fn_bytes, args, kwargs)`` and sends back
    ``(success, payload)`` where:

    - ``success`` is ``True`` → ``payload`` is return value
    - ``success`` is ``False`` → ``payload`` is an exception
    """
    try:
        while True:
            data = conn.recv()  # blocking
            if data == "STOP":
                return

            fn_bytes, args, kwargs = data

            try:
                fn = cloudpickle.loads(fn_bytes)
                result = fn(*args, **kwargs)
                conn.send((True, result))
            except Exception as exc:
                conn.send((False, exc))

    except EOFError:
        # Parent died or pipe closed
        return


# ===============================================================
# Persistent worker with lazy start
# ===============================================================
class Worker:
    """A persistent worker subprocess that executes cloudpickled functions."""

    def __init__(self, fn_bytes: bytes) -> None:
        """Initialize Worker with a cloudpickled function.

        Args:
            fn_bytes (bytes): Cloudpickled function to execute inside worker.
        """
        self.fn_bytes = fn_bytes
        self.conn: Connection | None = None
        self.proc: mp.Process | None = None

    # --------------------------
    # Worker lifecycle
    # --------------------------
    def ensure_started(self) -> None:
        """Start worker lazily if it's not running."""
        if self.proc is None or not self.proc.is_alive():
            self._start_worker()

    def _start_worker(self) -> None:
        """Create and start a worker subprocess."""
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()

        self.conn = parent_conn
        self.proc = ctx.Process(target=_worker_loop, args=(child_conn,))
        self.proc.start()

    def restart(self) -> None:
        """Restart the worker after timeout or crash."""
        self.kill()
        self._start_worker()

    def kill(self) -> None:
        """Terminate worker safely."""
        if self.conn is not None:
            try:  # noqa: SIM105
                self.conn.send("STOP")
            except Exception:  # noqa: BLE001, S110
                # Worker may already be dead or pipe closed
                pass

        if self.proc is not None:
            self.proc.terminate()
            self.proc.join()

        self.conn = None
        self.proc = None

    # --------------------------
    # Execute a function inside worker
    # --------------------------
    def call(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout: float,
    ) -> tuple[bool, Any]:
        """Execute the wrapped function inside worker with a timeout.

        Args:
            args (tuple[Any, ...]): Positional arguments for the function.
            kwargs (dict[str, Any]): Keyword arguments for the function.
            timeout (float): Timeout in seconds.

        Returns:
            tuple[bool, Any]: Tuple where first element indicates success,
            and second element is either the return value or an exception.

        Raises:
            TimeoutError: If the call times out.
        """
        self.ensure_started()
        assert self.conn is not None

        self.conn.send((self.fn_bytes, args, kwargs))

        if not self.conn.poll(timeout):
            msg = f"Worker timed out after {timeout} seconds"
            raise TimeoutError(msg)

        return self.conn.recv()


# ===============================================================
# Public timeout decorator
# ===============================================================
def timeout(
    seconds: float,
    default: R,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator applying a subprocess timeout to a function.

    Args:
        seconds (float): Timeout in seconds.
        default (R): Default return value on timeout.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: Decorated function.

    Raises:
        ValueError: If seconds is not > 0.
    """
    if seconds <= 0:
        msg = "seconds must be > 0"
        raise ValueError(msg)

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        # Use a wrapper instead of raw function to avoid re-import loops
        def safe_fn(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            return fn(*args, **kwargs)

        fn_bytes = cloudpickle.dumps(safe_fn)
        worker = Worker(fn_bytes)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                success, payload = worker.call(args, kwargs, timeout=seconds)

                if success:
                    return payload

                # Function raised inside worker
                raise payload  # noqa: TRY301

            except TimeoutError:
                worker.restart()
                return default

            except Exception:
                worker.restart()
                raise

        return wrapper

    return deco
