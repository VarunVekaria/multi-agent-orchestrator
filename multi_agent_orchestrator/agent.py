"""An autonomous agent that owns tasks and coordinates by asking its peers.

An agent has exactly two jobs:

1. **Serve** -- answer :meth:`Agent.get_task_status` for any task it owns, so
   other agents can find out whether a prerequisite is done.
2. **Run** -- work through its own tasks. Independent tasks start immediately.
   A dependent task polls whichever agent owns each prerequisite until every
   one of them reports success.

Nothing schedules an agent. It decides for itself when its work can begin.
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .errors import (
    DependencyFailedError,
    DependencyTimeoutError,
    UnknownDependencyError,
    UnknownTaskError,
)
from .events import EventBus, EventType
from .models import ExecutionContext, Handler, Task, TaskSpec, TaskStatus
from .registry import AgentRegistry
from .transport import LocalStatusClient, StatusClient


@dataclass(frozen=True)
class PollPolicy:
    """How aggressively an agent re-asks about an unfinished prerequisite.

    Backoff is not a nicety. Without it, every waiting task spins at the base
    interval for as long as its prerequisites take, so cost scales with the
    size of the graph rather than with the work being done. Jitter keeps agents
    that started together from settling into synchronised poll bursts.
    """

    base_interval: float = 0.05
    max_interval: float = 2.0
    multiplier: float = 1.6
    jitter: float = 0.25  # +/- fraction of the computed delay

    def delay_for(self, round_index: int, rng: random.Random) -> float:
        delay = min(
            self.base_interval * (self.multiplier**round_index), self.max_interval
        )
        if self.jitter:
            delay *= 1.0 + rng.uniform(-self.jitter, self.jitter)
        return max(0.0, delay)


class Agent:
    """Owns a set of tasks and drives them to completion."""

    def __init__(
        self,
        id: str,
        *,
        specs: Iterable[TaskSpec] = (),
        max_concurrency: int = 8,
        poll_policy: PollPolicy | None = None,
        registry: AgentRegistry | None = None,
        status_client: StatusClient | None = None,
        event_bus: EventBus | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.id = id
        self.max_concurrency = max_concurrency
        self.poll_policy = poll_policy or PollPolicy()
        self._tasks: dict[str, Task] = {}
        self._registry = registry
        self._status_client = status_client
        self._events = event_bus or EventBus()
        self._rng = rng or random.Random()
        self._semaphore: asyncio.Semaphore | None = None
        for spec in specs:
            self.add_task(spec)
        if registry is not None:
            registry.register(self)

    # -- task intake -----------------------------------------------------

    def add_task(self, spec: TaskSpec) -> Task:
        if spec.agent_id != self.id:
            raise ValueError(
                f"task {spec.id!r} is assigned to agent {spec.agent_id!r}, "
                f"not {self.id!r}"
            )
        if spec.id in self._tasks:
            raise ValueError(f"agent {self.id!r} already owns task {spec.id!r}")
        task = Task(spec=spec)
        self._tasks[spec.id] = task
        self._events.emit(
            EventType.TASK_SUBMITTED,
            self.id,
            spec.id,
            depends_on=list(spec.depends_on),
        )
        if self._registry is not None:
            self._registry.reindex(self)
        return task

    def add_tasks(self, specs: Iterable[TaskSpec]) -> list[Task]:
        return [self.add_task(spec) for spec in specs]

    def task(
        self,
        id: str,
        handler: Handler,
        *,
        depends_on: Sequence[str] = (),
        **kwargs: Any,
    ) -> Task:
        """Convenience wrapper that builds the :class:`TaskSpec` for you."""
        return self.add_task(
            TaskSpec(
                id=id,
                agent_id=self.id,
                handler=handler,
                depends_on=depends_on,
                **kwargs,
            )
        )

    def attach(
        self,
        *,
        registry: AgentRegistry | None = None,
        status_client: StatusClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Wire in shared infrastructure. Called by the orchestrator."""
        if registry is not None:
            self._registry = registry
        if status_client is not None:
            self._status_client = status_client
        if event_bus is not None:
            self._events = event_bus

    # -- introspection ---------------------------------------------------

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    @property
    def specs(self) -> tuple[TaskSpec, ...]:
        return tuple(task.spec for task in self._tasks.values())

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(self._tasks.values())

    @property
    def results(self) -> dict[str, Any]:
        return {
            task.id: task.result
            for task in self._tasks.values()
            if task.status.is_successful
        }

    @property
    def is_finished(self) -> bool:
        return all(task.status.is_terminal for task in self._tasks.values())

    # -- serving surface -------------------------------------------------

    def get_task_status(self, task_id: str):
        """Answer another agent's question about one of our tasks.

        This is the whole of the inter-agent protocol. It returns an immutable
        snapshot, never internal state, and never blocks -- a peer asking about
        our progress must not be able to stall us or corrupt us.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise UnknownTaskError(self.id, task_id)
        return task.snapshot()

    # -- execution -------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        """Drive every owned task to a terminal state.

        One coroutine per task, so a task blocked on a prerequisite never holds
        up a task that is ready to go.
        """
        if self._status_client is None:
            if self._registry is None:
                raise RuntimeError(
                    f"agent {self.id!r} has no registry or status client; "
                    "run it through an Orchestrator or call attach()"
                )
            self._status_client = LocalStatusClient(self._registry)

        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._events.emit(EventType.AGENT_STARTED, self.id, task_count=len(self._tasks))
        await asyncio.gather(*(self._run_task(task) for task in self._tasks.values()))
        self._events.emit(
            EventType.AGENT_FINISHED,
            self.id,
            completed=sum(1 for t in self._tasks.values() if t.status.is_successful),
            total=len(self._tasks),
        )
        return self.results

    async def _run_task(self, task: Task) -> None:
        try:
            deps = await self._await_dependencies(task)
        except DependencyFailedError as exc:
            # The prerequisite will never succeed, so neither can we. Marking
            # SKIPPED (terminal, unsuccessful) is what lets the cascade carry
            # itself downstream: our own dependents see this on their next poll
            # and skip in turn, with no propagation machinery involved.
            self._finish(
                task,
                TaskStatus.SKIPPED,
                error=str(exc),
                blocked_by=exc.dependency_id,
            )
            return
        except DependencyTimeoutError as exc:
            self._finish(task, TaskStatus.FAILED, error=str(exc))
            return

        self._set_status(task, TaskStatus.READY)
        assert self._semaphore is not None
        async with self._semaphore:
            await self._execute(task, deps)

    async def _await_dependencies(self, task: Task) -> dict[str, Any]:
        """Poll the owning agent of each prerequisite until all have settled.

        Returns the prerequisites' results, keyed by task id.
        """
        spec = task.spec
        if not spec.depends_on:
            return {}

        self._set_status(
            task, TaskStatus.WAITING, depends_on=list(spec.depends_on)
        )
        resolved: dict[str, Any] = {}
        pending = list(spec.depends_on)
        deadline = (
            time.monotonic() + spec.dependency_timeout
            if spec.dependency_timeout is not None
            else None
        )
        round_index = 0

        while pending:
            still_pending: list[str] = []
            for dependency_id in pending:
                response = await self._poll(task, dependency_id)
                if response.status.is_successful:
                    # Cached from here on: a settled prerequisite drops out of
                    # `pending` and is never polled again.
                    resolved[dependency_id] = response.result
                    self._events.emit(
                        EventType.DEPENDENCY_SATISFIED,
                        self.id,
                        task.id,
                        dependency=dependency_id,
                    )
                elif response.status.is_terminal:
                    self._events.emit(
                        EventType.DEPENDENCY_FAILED,
                        self.id,
                        task.id,
                        dependency=dependency_id,
                        status=str(response.status),
                    )
                    raise DependencyFailedError(
                        task.id, dependency_id, str(response.status)
                    )
                else:
                    still_pending.append(dependency_id)

            pending = still_pending
            if not pending:
                break

            delay = self.poll_policy.delay_for(round_index, self._rng)
            round_index += 1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DependencyTimeoutError(
                        task.id, pending, spec.dependency_timeout or 0.0
                    )
                delay = min(delay, remaining)
            await asyncio.sleep(delay)

        return resolved

    async def _poll(self, task: Task, dependency_id: str):
        """One status question about one prerequisite."""
        owner = self._registry.owner_of(dependency_id) if self._registry else None
        if owner is None:
            # Normally impossible: the orchestrator rejects unowned dependencies
            # before anything starts. Reachable only if an agent is driven by
            # hand, and worth failing loudly on rather than polling forever.
            raise UnknownDependencyError(task.id, dependency_id)

        self._events.emit(
            EventType.POLL_ISSUED,
            self.id,
            task.id,
            dependency=dependency_id,
            owner=owner,
        )
        assert self._status_client is not None
        return await self._status_client.get_status(owner, dependency_id)

    async def _execute(self, task: Task, deps: Mapping[str, Any]) -> None:
        spec = task.spec
        while True:
            task.attempts += 1
            task.started_at = time.monotonic()
            self._set_status(task, TaskStatus.RUNNING, attempt=task.attempts)
            context = ExecutionContext(
                task_id=task.id,
                agent_id=self.id,
                deps=dict(deps),
                attempt=task.attempts,
            )
            try:
                result = await self._invoke(spec.handler, context, spec.timeout)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                error = f"timed out after {spec.timeout}s"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                task.result = result
                self._finish(task, TaskStatus.COMPLETED)
                return

            if task.attempts <= spec.max_retries:
                self._events.emit(
                    EventType.TASK_RETRY,
                    self.id,
                    task.id,
                    attempt=task.attempts,
                    error=error,
                )
                await asyncio.sleep(spec.retry_backoff * task.attempts)
                continue

            self._finish(task, TaskStatus.FAILED, error=error)
            return

    @staticmethod
    async def _invoke(
        handler: Handler, context: ExecutionContext, timeout: float | None
    ) -> Any:
        """Call a handler, awaiting it if it is async.

        Sync handlers are supported for convenience, but they run on the event
        loop and so block every other agent for their duration -- and `timeout`
        cannot interrupt them. Anything slow should be async.
        """
        outcome = handler(context)
        if inspect.isawaitable(outcome):
            if timeout is not None:
                return await asyncio.wait_for(outcome, timeout)
            return await outcome
        return outcome

    # -- state transitions -----------------------------------------------

    _EVENT_FOR_STATUS = {
        TaskStatus.WAITING: EventType.TASK_WAITING,
        TaskStatus.READY: EventType.TASK_READY,
        TaskStatus.RUNNING: EventType.TASK_STARTED,
        TaskStatus.COMPLETED: EventType.TASK_COMPLETED,
        TaskStatus.FAILED: EventType.TASK_FAILED,
        TaskStatus.SKIPPED: EventType.TASK_SKIPPED,
    }

    def _set_status(self, task: Task, status: TaskStatus, **detail: Any) -> None:
        task.status = status
        event_type = self._EVENT_FOR_STATUS.get(status)
        if event_type is not None:
            self._events.emit(event_type, self.id, task.id, **detail)

    def _finish(
        self,
        task: Task,
        status: TaskStatus,
        *,
        error: str | None = None,
        blocked_by: str | None = None,
    ) -> None:
        task.error = error
        task.blocked_by = blocked_by
        task.finished_at = time.monotonic()
        detail: dict[str, Any] = {}
        if error is not None:
            detail["error"] = error
        if blocked_by is not None:
            detail["blocked_by"] = blocked_by
        if status is TaskStatus.COMPLETED and task.duration is not None:
            detail["duration"] = round(task.duration, 4)
        self._set_status(task, status, **detail)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Agent {self.id!r} tasks={len(self._tasks)}>"
