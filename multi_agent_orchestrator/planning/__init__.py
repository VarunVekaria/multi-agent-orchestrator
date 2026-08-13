"""Turning a natural-language query into agents and their tasks.

This is decomposition only. It decides *who* should exist and *what* each of
them would do -- it does not decide ordering, does not build ``TaskSpec``
objects, and does not run anything.

    from multi_agent_orchestrator.planning import Planner

    plan = await Planner().decompose("Research 3 web frameworks and compare")
    for agent in plan.agents:
        print(agent.id, [task.id for task in agent.tasks])
"""

from .errors import (
    PlannerRefusedError,
    PlannerUnavailableError,
    PlanningError,
)
from .llm import AnthropicLLM, LLMClient, ScriptedLLM
from .planner import DEFAULT_MODEL, Planner
from .schema import DecomposedPlan, PlannedAgent, PlannedTask

__all__ = [
    "AnthropicLLM",
    "DEFAULT_MODEL",
    "DecomposedPlan",
    "LLMClient",
    "PlannedAgent",
    "PlannedTask",
    "Planner",
    "PlannerRefusedError",
    "PlannerUnavailableError",
    "PlanningError",
    "ScriptedLLM",
]
