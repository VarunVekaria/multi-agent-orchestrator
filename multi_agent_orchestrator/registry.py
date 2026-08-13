"""Agent discovery.

The registry is the only piece of shared infrastructure the agents rely on. It
answers one question -- "which agent owns task X?" -- which is what lets a task
declare ``depends_on=["fetch_users"]`` without knowing or caring who produces
it. Everything else about coordination happens agent-to-agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import networkx as nx

from .errors import (
    DependencyCycleError,
    DuplicateTaskError,
    UnknownAgentError,
    UnknownDependencyError,
)
from .models import TaskSpec

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .agent import Agent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, "Agent"] = {}
        self._task_owners: dict[str, str] = {}

    # -- registration ----------------------------------------------------

    def register(self, agent: "Agent") -> None:
        """Add an agent and index every task it owns.

        Safe to call again after an agent is handed more tasks; the task index
        is refreshed each time.
        """
        existing = self._agents.get(agent.id)
        if existing is not None and existing is not agent:
            raise ValueError(
                f"a different agent is already registered as {agent.id!r}"
            )
        self._agents[agent.id] = agent
        self.reindex(agent)

    def reindex(self, agent: "Agent") -> None:
        for task_id in agent.task_ids:
            owner = self._task_owners.get(task_id)
            if owner is not None and owner != agent.id:
                raise DuplicateTaskError(task_id, owner, agent.id)
            self._task_owners[task_id] = agent.id

    # -- lookup ----------------------------------------------------------

    def get_agent(self, agent_id: str) -> "Agent":
        try:
            return self._agents[agent_id]
        except KeyError:
            raise UnknownAgentError(agent_id) from None

    def owner_of(self, task_id: str) -> str | None:
        """Agent id that owns ``task_id``, or None if nobody claims it."""
        return self._task_owners.get(task_id)

    def require_owner_of(self, task_id: str) -> str:
        owner = self.owner_of(task_id)
        if owner is None:
            raise UnknownAgentError(f"<owner of task {task_id!r}>")
        return owner

    @property
    def agents(self) -> tuple["Agent", ...]:
        return tuple(self._agents.values())

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(self._agents)

    def specs(self) -> Iterator[TaskSpec]:
        for agent in self._agents.values():
            yield from agent.specs

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    # -- validation ------------------------------------------------------

    def build_graph(self) -> nx.DiGraph:
        """Dependency graph over task ids, edge = prerequisite -> dependent."""
        graph = nx.DiGraph()
        for spec in self.specs():
            graph.add_node(spec.id, agent_id=spec.agent_id)
        for spec in self.specs():
            for dependency in spec.depends_on:
                graph.add_edge(dependency, spec.id)
        return graph

    def validate(self) -> nx.DiGraph:
        """Reject any plan that would leave an agent polling forever.

        Two structural faults do that, and both are cheap to catch here rather
        than discovering them as a hung process:

        * a dependency nobody owns -- the poll never gets a terminal answer
        * a cycle -- every agent in it waits on the next, indefinitely
        """
        for spec in self.specs():
            for dependency in spec.depends_on:
                if dependency not in self._task_owners:
                    raise UnknownDependencyError(spec.id, dependency)

        graph = self.build_graph()
        if not nx.is_directed_acyclic_graph(graph):
            cycles = sorted(nx.simple_cycles(graph), key=len)
            raise DependencyCycleError(cycles)
        return graph
