"""Structured events emitted as agents work.

These drive the demo's live table, but their more important job is making the
system testable: assertions about ordering ("did `report` start only after
`enrich` completed?") read off the event log instead of off wall-clock sleeps,
which keeps the test suite deterministic on a loaded machine.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


class EventType(str, enum.Enum):
    AGENT_STARTED = "agent_started"
    AGENT_FINISHED = "agent_finished"
    TASK_SUBMITTED = "task_submitted"
    TASK_WAITING = "task_waiting"
    POLL_ISSUED = "poll_issued"
    DEPENDENCY_SATISFIED = "dependency_satisfied"
    DEPENDENCY_FAILED = "dependency_failed"
    TASK_READY = "task_ready"
    TASK_STARTED = "task_started"
    TASK_RETRY = "task_retry"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Event:
    type: EventType
    agent_id: str
    task_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.monotonic)


Subscriber = Callable[[Event], None]


class EventBus:
    """Synchronous fan-out to subscribers, with optional in-memory history."""

    def __init__(self, *, record_history: bool = True) -> None:
        self._subscribers: list[Subscriber] = []
        self._history: list[Event] = []
        self._record_history = record_history

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def emit(
        self,
        type: EventType,
        agent_id: str,
        task_id: str | None = None,
        **detail: Any,
    ) -> Event:
        event = Event(type=type, agent_id=agent_id, task_id=task_id, detail=detail)
        if self._record_history:
            self._history.append(event)
        for subscriber in self._subscribers:
            # A broken subscriber (say, a rendering glitch in the demo UI) must
            # never take down the agent that was only trying to report progress.
            try:
                subscriber(event)
            except Exception:
                logger.exception("event subscriber raised for %s", event.type)
        return event

    @property
    def history(self) -> list[Event]:
        return list(self._history)

    def of_type(self, *types: EventType) -> list[Event]:
        wanted = set(types)
        return [e for e in self._history if e.type in wanted]

    def for_task(self, task_id: str) -> list[Event]:
        return [e for e in self._history if e.task_id == task_id]

    def order_of(self, *types: EventType) -> list[tuple[EventType, str | None]]:
        """Sequence of (type, task_id) for the given types, in emission order."""
        wanted = set(types)
        return [(e.type, e.task_id) for e in self._history if e.type in wanted]

    def index_of(self, type: EventType, task_id: str) -> int:
        """Position of a specific event in history; -1 if it never happened."""
        for i, event in enumerate(self._history):
            if event.type is type and event.task_id == task_id:
                return i
        return -1

    def extend(self, events: Iterable[Event]) -> None:
        if self._record_history:
            self._history.extend(events)

    def clear(self) -> None:
        self._history.clear()
