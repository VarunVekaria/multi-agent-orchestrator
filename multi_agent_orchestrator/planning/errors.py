"""Failures specific to decomposition.

All of these subclass :class:`~multi_agent_orchestrator.errors.OrchestratorError`
so a caller can catch one family across planning and execution.
"""

from __future__ import annotations

from ..errors import OrchestratorError


class PlanningError(OrchestratorError):
    """Base for anything that goes wrong turning a query into a plan."""


class PlannerUnavailableError(PlanningError):
    """The planner could not be reached or was not configured.

    Raised for a missing API key rather than letting the SDK surface its own
    error, because "no credentials" and "the model failed" are different
    problems and only one of them is the caller's to fix.
    """


class PlannerRefusedError(PlanningError):
    """The model declined the query.

    A refusal arrives as a normal HTTP 200 with ``stop_reason == "refusal"``,
    so it has to be checked for explicitly -- reading the response content
    without checking would raise a confusing IndexError instead.
    """

    def __init__(self, category: str | None = None, explanation: str | None = None):
        self.category = category
        self.explanation = explanation
        detail = f" ({category})" if category else ""
        message = explanation or "the model declined to plan this query"
        super().__init__(f"planner refused{detail}: {message}")


class InvalidPlanError(PlanningError):
    """The model returned a plan that cannot be used.

    Distinct from a schema failure: the shape was right but the content was
    not, e.g. two tasks sharing an id.
    """
