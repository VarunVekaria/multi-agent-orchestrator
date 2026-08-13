"""How a status question travels from one agent to another.

The polling code in :mod:`.agent` talks to this interface and nothing else, so
it has no idea whether the agent it is questioning lives in the same process or
across a network. Today there is one implementation, backed by the registry. A
network transport would be a second class here and would change no other file.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import TaskStatusResponse
from .registry import AgentRegistry


@runtime_checkable
class StatusClient(Protocol):
    """Asks an agent whether one of its tasks has finished."""

    async def get_status(self, agent_id: str, task_id: str) -> TaskStatusResponse:
        ...


class LocalStatusClient:
    """In-process transport: resolve the agent and ask it directly."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    async def get_status(self, agent_id: str, task_id: str) -> TaskStatusResponse:
        agent = self._registry.get_agent(agent_id)
        return agent.get_task_status(task_id)
