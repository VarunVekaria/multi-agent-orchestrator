"""The shape the planner must return.

These models are handed straight to the Anthropic SDK as a structured-output
format, so they double as the prompt contract: the field descriptions are what
the model reads to decide what to put where. Keep them flat -- structured
outputs reject recursive schemas.

Deliberately absent: any notion of ordering or prerequisites. This step decides
who exists and what they would do, nothing about when.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .errors import InvalidPlanError

_SLUG_STRIP = re.compile(r"[^a-z0-9_]+")
_SLUG_COLLAPSE = re.compile(r"_{2,}")


def slugify(value: str) -> str:
    """Normalise a model-supplied id into ``lower_snake_case``.

    The model is asked for ids in this form and usually complies, but "Fetch
    Data" or "fetch-data" show up often enough that rejecting them would fail
    the whole plan over cosmetics. Normalising is forgiving where it costs
    nothing; genuine problems like duplicates are still errors.
    """
    slug = _SLUG_STRIP.sub("_", value.strip().lower())
    slug = _SLUG_COLLAPSE.sub("_", slug).strip("_")
    return slug


class _Slugged(BaseModel):
    @field_validator("id", check_fields=False)
    @classmethod
    def _normalise_id(cls, value: str) -> str:
        slug = slugify(value)
        if not slug:
            raise ValueError(f"id {value!r} has no usable characters")
        return slug


class PlannedTask(_Slugged):
    """One unit of work belonging to a single agent."""

    id: str = Field(
        description="Short lower_snake_case identifier, unique across the whole "
        "plan, e.g. 'fetch_pricing_data'."
    )
    description: str = Field(
        min_length=1,
        description="What this task produces, in one or two sentences. Written "
        "so it could be handed to someone with no other context.",
    )


class PlannedAgent(_Slugged):
    """A specialist that owns some tasks."""

    id: str = Field(
        description="Short lower_snake_case identifier, unique across the plan, "
        "e.g. 'researcher'."
    )
    role: str = Field(
        min_length=1,
        description="What this agent specialises in and is responsible for, in "
        "one or two sentences.",
    )
    tasks: list[PlannedTask] = Field(
        min_length=1,
        description="The tasks this agent owns. Split the agent's work into "
        "several tasks where it genuinely decomposes; use a single task where "
        "it does not. Do not invent filler tasks to pad the list.",
    )


class DecomposedPlan(BaseModel):
    """The full answer: which agents exist, and what each of them would do."""

    agents: list[PlannedAgent] = Field(
        min_length=1,
        description="The agents needed for this query. Use as many as the work "
        "genuinely calls for and no more.",
    )

    @model_validator(mode="after")
    def _ids_must_be_unique(self) -> "DecomposedPlan":
        """Reject id collisions.

        Task ids have to be unique across the *entire* plan, not just within an
        agent: the registry indexes tasks by id alone so that a dependency can
        name a task without knowing its owner, and it raises DuplicateTaskError
        on a collision. Catching that here means a bad plan fails at the point
        it was produced rather than much later at submission.
        """
        _reject_duplicates([agent.id for agent in self.agents], "agent")
        _reject_duplicates(
            [task.id for agent in self.agents for task in agent.tasks], "task"
        )
        return self

    @property
    def task_count(self) -> int:
        return sum(len(agent.tasks) for agent in self.agents)

    def describe(self) -> str:
        """Plain-text rendering, for CLI output and test failure messages."""
        lines = [
            f"{len(self.agents)} agents, {self.task_count} tasks",
        ]
        for agent in self.agents:
            lines.append(f"\n  {agent.id} -- {agent.role}")
            for task in agent.tasks:
                lines.append(f"    - {task.id}: {task.description}")
        return "\n".join(lines)


def _reject_duplicates(ids: list[str], kind: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for id in ids:
        if id in seen and id not in duplicates:
            duplicates.append(id)
        seen.add(id)
    if duplicates:
        raise InvalidPlanError(
            f"duplicate {kind} id(s): {', '.join(sorted(duplicates))}"
        )


def plan_json_schema() -> dict[str, Any]:
    """The JSON Schema sent to the model. Exposed for tests and debugging."""
    return DecomposedPlan.model_json_schema()
