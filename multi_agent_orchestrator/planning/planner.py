"""Query in, agents and their tasks out.

The planner does one thing: decide who should exist and what each of them would
do. It does not order the work, does not build ``TaskSpec`` objects, and does
not execute anything.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .llm import AnthropicLLM, LLMClient
from .schema import DecomposedPlan

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"

#: The whole prompt contract lives here plus the field descriptions in
#: schema.py. It is deliberately about *how to divide work*, not about output
#: format -- structured outputs already guarantee the shape, so spending prompt
#: on "return valid JSON" would be wasted and is a dated habit.
SYSTEM_PROMPT = """\
You break a request into a team of agents and the tasks each agent owns.

An agent is a specialist with a clear remit. A task is one concrete piece of \
work that produces something.

How to decide:

- Pick the number of agents the work genuinely calls for. A narrow request may \
need one. A broad one may need several. Do not pad the team to look thorough, \
and do not force unrelated work into a single agent to look lean.
- Separate agents by the kind of work they do, not by the order it happens in. \
"Researcher" and "writer" are different specialists; "step one" and "step two" \
are not.
- Give an agent several tasks where its work genuinely divides, and a single \
task where it does not. An agent with one real task is a fine answer.
- Write each task so someone could act on it without seeing this conversation \
or the other tasks.

Do not describe ordering, prerequisites, or which task must finish before \
another. That is decided elsewhere. Describe only who exists and what each of \
them does."""


@dataclass
class Planner:
    """Decomposes a query into agents and tasks.

    ``llm`` defaults to the real Claude client; pass a
    :class:`~multi_agent_orchestrator.planning.llm.ScriptedLLM` to run offline.
    """

    llm: LLMClient = field(default_factory=AnthropicLLM)
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    system_prompt: str = SYSTEM_PROMPT

    async def decompose(self, query: str) -> DecomposedPlan:
        """Break ``query`` into agents and their tasks.

        Raises :class:`PlannerUnavailableError` if the model cannot be reached,
        :class:`PlannerRefusedError` if it declines, and
        :class:`InvalidPlanError` if the plan it returns is internally
        inconsistent (duplicate ids).
        """
        query = query.strip()
        if not query:
            raise ValueError("query must contain non-whitespace characters")

        return await self.llm.parse_plan(
            system=self.system_prompt,
            query=query,
            model=self.model,
            effort=self.effort,
        )

    def decompose_sync(self, query: str) -> DecomposedPlan:
        """Blocking entry point, for scripts and the CLI."""
        return asyncio.run(self.decompose(query))
