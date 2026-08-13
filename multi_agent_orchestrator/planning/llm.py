"""The seam between the planner and whatever answers it.

Mirrors the ``StatusClient`` split in :mod:`multi_agent_orchestrator.transport`:
the planner talks to a protocol, so the real model and a scripted stand-in are
interchangeable. That is what makes decomposition testable without a network
call, an API key, or a bill.

Everything here is async. A synchronous SDK call would block the event loop the
agents share, and this module is meant to be called from the same loop they run
on.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

from .errors import PlannerRefusedError, PlannerUnavailableError
from .schema import DecomposedPlan


class LLMClient(Protocol):
    """Anything that can turn a prompt into a :class:`DecomposedPlan`."""

    async def parse_plan(
        self, *, system: str, query: str, model: str, effort: str
    ) -> DecomposedPlan: ...


class AnthropicLLM:
    """Backed by the real Claude API via the async SDK."""

    def __init__(self, api_key: str | None = None, max_tokens: int = 8_000) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._max_tokens = max_tokens
        self._client = None  # built lazily so importing this module is free

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise PlannerUnavailableError(
                "ANTHROPIC_API_KEY is not set. Export it, or pass a different "
                "LLMClient to Planner(llm=...) -- ScriptedLLM needs no key."
            )

        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise PlannerUnavailableError(
                "the anthropic package is not installed; "
                'install it with: pip install -e ".[agentic]"'
            ) from exc

        self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def parse_plan(
        self, *, system: str, query: str, model: str, effort: str
    ) -> DecomposedPlan:
        client = self._ensure_client()

        response = await client.messages.parse(
            model=model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": query}],
            output_format=DecomposedPlan,
            # Decomposition is a judgement call about how work divides up, which
            # is exactly the kind of thing thinking helps with.
            thinking={"type": "adaptive"},
            # The SDK merges `format` into this dict, so effort survives.
            output_config={"effort": effort},
        )

        # Checked before touching the response: a refusal is a 200 with empty
        # content, so reading parsed_output first would raise something
        # unrelated and hide the real reason.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise PlannerRefusedError(
                category=getattr(details, "category", None),
                explanation=getattr(details, "explanation", None),
            )

        return response.parsed_output


class ScriptedLLM:
    """Returns pre-built plans in order. For tests and offline demos.

    Takes plans directly rather than raw JSON so a test states the plan it
    means; malformed-response handling belongs to the schema's own tests.
    """

    def __init__(self, plans: Sequence[DecomposedPlan | Exception]) -> None:
        self._plans = list(plans)
        self.calls: list[dict[str, str]] = []

    async def parse_plan(
        self, *, system: str, query: str, model: str, effort: str
    ) -> DecomposedPlan:
        # Recorded so tests can assert on what the planner actually sent.
        self.calls.append(
            {"system": system, "query": query, "model": model, "effort": effort}
        )

        if not self._plans:
            raise AssertionError(
                f"ScriptedLLM ran out of scripted plans on call {len(self.calls)}"
            )

        nxt = self._plans.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt
