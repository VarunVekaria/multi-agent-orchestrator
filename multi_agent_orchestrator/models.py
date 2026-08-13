"""Core data types shared by agents, the registry and the transport layer."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union


class TaskStatus(str, enum.Enum):
    """Lifecycle of a single task.

    PENDING -> WAITING -> READY -> RUNNING -> {COMPLETED, FAILED, SKIPPED}

    A task with no dependencies skips WAITING entirely. SKIPPED means a
    prerequisite did not succeed, so this task will never run.
    """

    PENDING = "PENDING"
    WAITING = "WAITING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES

    @property
    def is_successful(self) -> bool:
        return self is TaskStatus.COMPLETED

    def __str__(self) -> str:  # keeps rich tables and log lines readable
        return self.value


_TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
)


@dataclass(frozen=True)
class ExecutionContext:
    """Handed to a task handler when it runs.

    ``deps`` carries the results of every prerequisite, keyed by task id, so a
    dependent task can consume upstream output without reaching into any agent.
    """

    task_id: str
    agent_id: str
    deps: Mapping[str, Any] = field(default_factory=dict)
    attempt: int = 1


# Handlers may be async or plain sync callables; the agent normalises both.
Handler = Callable[[ExecutionContext], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class TaskSpec:
    """Declarative description of a unit of work assigned to one agent."""

    id: str
    agent_id: str
    handler: Handler
    depends_on: Sequence[str] = ()
    #: Wall-clock budget for a single execution attempt.
    timeout: float | None = None
    #: Wall-clock budget for waiting on prerequisites before giving up.
    dependency_timeout: float | None = None
    max_retries: int = 0
    retry_backoff: float = 0.1

    def __post_init__(self) -> None:
        # Normalising to a tuple keeps the spec hashable and stops a caller's
        # later mutation of the list from changing the dependency graph after
        # validation has already run.
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if self.max_retries < 0:
            raise ValueError(f"task {self.id!r}: max_retries must be >= 0")
        if self.id in self.depends_on:
            raise ValueError(f"task {self.id!r} depends on itself")


@dataclass
class Task:
    """Runtime state for a spec. Owned exclusively by one agent."""

    spec: TaskSpec
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    #: Set when the task is SKIPPED: the prerequisite that did not succeed.
    blocked_by: str | None = None
    attempts: int = 0
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def agent_id(self) -> str:
        return self.spec.agent_id

    @property
    def duration(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    def snapshot(self) -> "TaskStatusResponse":
        """Build the read-only view other agents are allowed to see."""
        return TaskStatusResponse(
            task_id=self.id,
            agent_id=self.agent_id,
            status=self.status,
            result=self.result if self.status.is_successful else None,
            error=self.error,
            blocked_by=self.blocked_by,
        )


@dataclass(frozen=True)
class TaskStatusResponse:
    """The only thing one agent ever gets back from another.

    This is the wire contract. It is frozen so a polling agent cannot reach
    through the response and mutate the owning agent's task state; when this
    system grows a network transport, this is the type that gets serialised.
    """

    task_id: str
    agent_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    blocked_by: str | None = None
