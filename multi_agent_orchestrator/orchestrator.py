"""Assembly and lifecycle, deliberately not scheduling.

The orchestrator builds the registry, wires every agent to the shared transport
and event bus, validates the plan, and starts all the agents at once. It never
decides what runs when -- once :meth:`Orchestrator.run` is called, every agent
is on its own, and the execution order is an emergent property of the tasks'
dependencies.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import networkx as nx

from .agent import Agent, PollPolicy
from .events import EventBus
from .models import Task, TaskSpec, TaskStatus
from .registry import AgentRegistry
from .transport import LocalStatusClient, StatusClient


@dataclass
class RunReport:
    """Everything that happened, once every agent has gone quiet."""

    tasks: dict[str, Task]
    duration: float
    events: EventBus

    @property
    def results(self) -> dict[str, Any]:
        return {
            id: task.result
            for id, task in self.tasks.items()
            if task.status.is_successful
        }

    def by_status(self, status: TaskStatus) -> list[Task]:
        return [t for t in self.tasks.values() if t.status is status]

    @property
    def completed(self) -> list[Task]:
        return self.by_status(TaskStatus.COMPLETED)

    @property
    def failed(self) -> list[Task]:
        return self.by_status(TaskStatus.FAILED)

    @property
    def skipped(self) -> list[Task]:
        return self.by_status(TaskStatus.SKIPPED)

    @property
    def succeeded(self) -> bool:
        """True when every task completed."""
        return all(t.status.is_successful for t in self.tasks.values())

    def status_of(self, task_id: str) -> TaskStatus:
        return self.tasks[task_id].status

    def result_of(self, task_id: str) -> Any:
        return self.tasks[task_id].result

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for task in self.tasks.values():
            counts[str(task.status)] = counts.get(str(task.status), 0) + 1
        rendered = ", ".join(f"{n} {s.lower()}" for s, n in sorted(counts.items()))
        return f"{len(self.tasks)} tasks in {self.duration:.2f}s: {rendered}"


@dataclass
class Orchestrator:
    poll_policy: PollPolicy = field(default_factory=PollPolicy)
    events: EventBus = field(default_factory=EventBus)
    max_concurrency: int = 8
    rng: random.Random | None = None
    registry: AgentRegistry = field(default_factory=AgentRegistry)
    status_client: StatusClient | None = None

    def __post_init__(self) -> None:
        if self.status_client is None:
            self.status_client = LocalStatusClient(self.registry)

    # -- assembly --------------------------------------------------------

    def add_agent(self, agent: Agent) -> Agent:
        agent.attach(
            registry=self.registry,
            status_client=self.status_client,
            event_bus=self.events,
        )
        self.registry.register(agent)
        return agent

    def agent(self, agent_id: str, **kwargs: Any) -> Agent:
        """Get the agent with this id, creating it on first mention."""
        if agent_id in self.registry:
            return self.registry.get_agent(agent_id)
        agent = Agent(
            agent_id,
            max_concurrency=kwargs.pop("max_concurrency", self.max_concurrency),
            poll_policy=kwargs.pop("poll_policy", self.poll_policy),
            rng=kwargs.pop("rng", self.rng),
            **kwargs,
        )
        return self.add_agent(agent)

    def submit(self, spec: TaskSpec) -> Task:
        """Assign a task to its agent, creating that agent if it is new."""
        return self.agent(spec.agent_id).add_task(spec)

    def submit_all(self, specs: Iterable[TaskSpec]) -> list[Task]:
        return [self.submit(spec) for spec in specs]

    # -- inspection ------------------------------------------------------

    @property
    def tasks(self) -> dict[str, Task]:
        return {
            task.id: task for agent in self.registry.agents for task in agent.tasks
        }

    def validate(self) -> nx.DiGraph:
        """Reject unrunnable plans before a single agent starts.

        See :meth:`AgentRegistry.validate` -- this catches the two faults that
        would otherwise surface as a hang rather than an error.
        """
        return self.registry.validate()

    def dependency_graph(self) -> nx.DiGraph:
        return self.registry.build_graph()

    def execution_layers(self) -> list[list[str]]:
        """Task ids grouped into topological layers.

        Purely descriptive -- useful for rendering and for reasoning about a
        plan. Nothing in the runtime consults it; the agents work this out
        among themselves by polling.
        """
        return [sorted(layer) for layer in nx.topological_generations(self.validate())]

    # -- execution -------------------------------------------------------

    async def run(self) -> RunReport:
        """Validate, release every agent at once, and wait for quiet."""
        self.validate()
        started = time.monotonic()
        await asyncio.gather(*(agent.run() for agent in self.registry.agents))
        return RunReport(
            tasks=self.tasks,
            duration=time.monotonic() - started,
            events=self.events,
        )

    def run_sync(self) -> RunReport:
        """Blocking entry point for scripts and the CLI."""
        return asyncio.run(self.run())


def build(specs: Sequence[TaskSpec], **kwargs: Any) -> Orchestrator:
    """Shorthand: an orchestrator with every spec submitted and validated."""
    orchestrator = Orchestrator(**kwargs)
    orchestrator.submit_all(specs)
    orchestrator.validate()
    return orchestrator
