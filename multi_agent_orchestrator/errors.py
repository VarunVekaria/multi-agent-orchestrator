"""Exception hierarchy.

The validation errors matter more than usual here: because agents wait by
polling, a dependency that can never resolve is an infinite wait rather than a
crash. Every such case is detected before any agent starts.
"""

from __future__ import annotations

from typing import Sequence


class OrchestratorError(Exception):
    """Base class for every error raised by this package."""


class ValidationError(OrchestratorError):
    """Raised while validating a plan, before execution begins."""


class DependencyCycleError(ValidationError):
    """A set of tasks depends on itself transitively.

    Left unchecked this deadlocks every agent in the cycle, each politely
    polling the next forever.
    """

    def __init__(self, cycles: Sequence[Sequence[str]]) -> None:
        self.cycles = [list(cycle) for cycle in cycles]
        rendered = "; ".join(" -> ".join(c + [c[0]]) for c in self.cycles)
        super().__init__(f"dependency cycle detected: {rendered}")


class UnknownDependencyError(ValidationError):
    """A task depends on an id no registered agent owns."""

    def __init__(self, task_id: str, missing: str) -> None:
        self.task_id = task_id
        self.missing = missing
        super().__init__(
            f"task {task_id!r} depends on {missing!r}, which no agent owns"
        )


class DuplicateTaskError(ValidationError):
    """Two agents claim the same task id.

    Task ids are the routing key for status lookups, so they must be unique
    across the whole system, not just within one agent.
    """

    def __init__(self, task_id: str, first_agent: str, second_agent: str) -> None:
        self.task_id = task_id
        self.first_agent = first_agent
        self.second_agent = second_agent
        super().__init__(
            f"task id {task_id!r} is claimed by both agent {first_agent!r} "
            f"and agent {second_agent!r}"
        )


class UnknownAgentError(OrchestratorError):
    """A status lookup named an agent that is not registered."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        super().__init__(f"no agent registered with id {agent_id!r}")


class UnknownTaskError(OrchestratorError):
    """An agent was asked about a task it does not own."""

    def __init__(self, agent_id: str, task_id: str) -> None:
        self.agent_id = agent_id
        self.task_id = task_id
        super().__init__(f"agent {agent_id!r} does not own task {task_id!r}")


class DependencyFailedError(OrchestratorError):
    """A prerequisite reached a terminal state that was not COMPLETED."""

    def __init__(self, task_id: str, dependency_id: str, status: str) -> None:
        self.task_id = task_id
        self.dependency_id = dependency_id
        self.status = status
        super().__init__(
            f"task {task_id!r} cannot run: dependency {dependency_id!r} is {status}"
        )


class DependencyTimeoutError(OrchestratorError):
    """Prerequisites did not settle within the task's dependency budget."""

    def __init__(self, task_id: str, pending: Sequence[str], timeout: float) -> None:
        self.task_id = task_id
        self.pending = list(pending)
        self.timeout = timeout
        super().__init__(
            f"task {task_id!r} waited {timeout}s for {', '.join(self.pending)} "
            "without them settling"
        )
