"""Ready-made handlers.

The orchestration core is domain-agnostic: a task is just an id, an owner and a
callable. These handlers exist so the coordination behaviour can be exercised
and demonstrated without dragging in real I/O.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping

from .models import ExecutionContext, Handler


def produces(value: Any, *, duration: float = 0.1) -> Handler:
    """Do nothing for `duration`, then return `value`. The stand-in for I/O."""

    async def handler(context: ExecutionContext) -> Any:
        await asyncio.sleep(duration)
        return value

    return handler


def combines(
    fn: Callable[[Mapping[str, Any]], Any], *, duration: float = 0.1
) -> Handler:
    """Derive a result from upstream results.

    `fn` receives ``context.deps`` -- the results of this task's prerequisites,
    keyed by task id.
    """

    async def handler(context: ExecutionContext) -> Any:
        await asyncio.sleep(duration)
        return fn(context.deps)

    return handler


def always_fails(message: str = "worker failed", *, duration: float = 0.05) -> Handler:
    """Raise every time. Used to show the SKIPPED cascade downstream."""

    async def handler(context: ExecutionContext) -> Any:
        await asyncio.sleep(duration)
        raise RuntimeError(f"{context.task_id}: {message}")

    return handler


def flaky(*, succeeds_on_attempt: int = 2, value: Any = "ok", duration: float = 0.05) -> Handler:
    """Fail until the given attempt number, then succeed. Exercises retries."""

    async def handler(context: ExecutionContext) -> Any:
        await asyncio.sleep(duration)
        if context.attempt < succeeds_on_attempt:
            raise RuntimeError(
                f"{context.task_id}: attempt {context.attempt} failed"
            )
        return value

    return handler


def never_finishes() -> Handler:
    """Hang forever. Used to prove `timeout` and `dependency_timeout` bite."""

    async def handler(context: ExecutionContext) -> Any:
        await asyncio.Event().wait()

    return handler
