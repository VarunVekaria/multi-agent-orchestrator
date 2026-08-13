"""Decomposition: query in, agents and their tasks out.

Every test here runs offline against ScriptedLLM. The one thing they cannot
cover is whether the real model returns a *sensible* plan -- only that whatever
it returns is handled correctly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multi_agent_orchestrator.planning import (
    DecomposedPlan,
    Planner,
    PlannerRefusedError,
    PlannerUnavailableError,
    ScriptedLLM,
)
from multi_agent_orchestrator.planning.errors import InvalidPlanError
from multi_agent_orchestrator.planning.llm import AnthropicLLM
from multi_agent_orchestrator.planning.schema import slugify


def plan(**agents: dict[str, str]) -> DecomposedPlan:
    """Build a plan from ``agent_id={task_id: description}`` for brevity."""
    return DecomposedPlan(
        agents=[
            {
                "id": agent_id,
                "role": f"{agent_id} specialist",
                "tasks": [
                    {"id": task_id, "description": description}
                    for task_id, description in tasks.items()
                ],
            }
            for agent_id, tasks in agents.items()
        ]
    )


# -- the core behaviour ----------------------------------------------------


async def test_decompose_returns_agents_each_with_their_tasks():
    expected = plan(
        researcher={
            "gather_sources": "Collect primary sources.",
            "extract_claims": "Pull the key claims out of each source.",
        },
        writer={"draft_report": "Write the comparison."},
    )
    llm = ScriptedLLM([expected])

    result = await Planner(llm=llm).decompose("Compare three web frameworks")

    assert [agent.id for agent in result.agents] == ["researcher", "writer"]
    assert [task.id for task in result.agents[0].tasks] == [
        "gather_sources",
        "extract_claims",
    ]
    # An agent whose work does not divide gets a single task, not filler.
    assert [task.id for task in result.agents[1].tasks] == ["draft_report"]
    assert result.task_count == 3


async def test_query_and_system_prompt_reach_the_model():
    llm = ScriptedLLM([plan(solo={"do_it": "Do the thing."})])

    await Planner(llm=llm, model="test-model", effort="low").decompose(
        "  Build me a dashboard  "
    )

    (call,) = llm.calls
    assert call["query"] == "Build me a dashboard", "query should be stripped"
    assert call["model"] == "test-model"
    assert call["effort"] == "low"
    # The prompt has to actually carry the decomposition instructions.
    assert "agent" in call["system"].lower()


async def test_agent_count_is_left_to_the_model():
    """One agent and many are both valid answers; nothing clamps the count."""
    for size in (1, 5):
        agents = {f"agent_{i}": {f"task_{i}": "Work."} for i in range(size)}
        result = await Planner(llm=ScriptedLLM([plan(**agents)])).decompose("q")
        assert len(result.agents) == size


async def test_blank_query_is_rejected_before_calling_the_model():
    llm = ScriptedLLM([plan(a={"t": "x"})])

    with pytest.raises(ValueError, match="non-whitespace"):
        await Planner(llm=llm).decompose("   \n  ")

    assert llm.calls == [], "should not spend a request on an empty query"


# -- schema rules ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Fetch Data", "fetch_data"),
        ("fetch-data", "fetch_data"),
        ("  Fetch   Data  ", "fetch_data"),
        ("Fetch/Data!", "fetch_data"),
        ("already_fine", "already_fine"),
    ],
)
def test_ids_are_normalised_rather_than_rejected(raw: str, expected: str):
    """Cosmetic id variation should not fail an otherwise good plan."""
    assert slugify(raw) == expected
    built = plan(**{"Some Agent": {raw: "Work."}})
    assert built.agents[0].id == "some_agent"
    assert built.agents[0].tasks[0].id == expected


def test_id_with_no_usable_characters_is_rejected():
    with pytest.raises(ValidationError, match="no usable characters"):
        plan(**{"!!!": {"t": "x"}})


def test_duplicate_task_ids_across_different_agents_are_rejected():
    """The registry indexes tasks by id alone, so plan-wide collisions break it."""
    with pytest.raises(InvalidPlanError, match="duplicate task id"):
        plan(
            researcher={"shared_id": "One."},
            writer={"shared_id": "Two."},
        )


def test_duplicate_agent_ids_are_rejected():
    with pytest.raises(InvalidPlanError, match="duplicate agent id"):
        DecomposedPlan(
            agents=[
                {"id": "dup", "role": "r", "tasks": [{"id": "a", "description": "x"}]},
                {"id": "dup", "role": "r", "tasks": [{"id": "b", "description": "y"}]},
            ]
        )


def test_an_agent_must_own_at_least_one_task():
    with pytest.raises(ValidationError):
        DecomposedPlan(agents=[{"id": "idle", "role": "r", "tasks": []}])


def test_a_plan_must_have_at_least_one_agent():
    with pytest.raises(ValidationError):
        DecomposedPlan(agents=[])


def test_schema_sent_to_the_model_is_not_recursive():
    """Structured outputs reject recursive schemas -- guard against a future edit."""
    from multi_agent_orchestrator.planning.schema import plan_json_schema

    schema = plan_json_schema()
    assert "$defs" in schema
    assert "PlannedAgent" in schema["$defs"] and "PlannedTask" in schema["$defs"]
    # A task must not be able to contain agents or tasks.
    assert set(schema["$defs"]["PlannedTask"]["properties"]) == {"id", "description"}


# -- failure paths ---------------------------------------------------------


async def test_refusal_surfaces_as_planner_refused():
    llm = ScriptedLLM([PlannerRefusedError(category="cyber", explanation="declined")])

    with pytest.raises(PlannerRefusedError) as caught:
        await Planner(llm=llm).decompose("something disallowed")

    assert caught.value.category == "cyber"


async def test_missing_api_key_names_the_variable():
    """The failure should point at the fix, not surface an SDK auth error."""
    planner = Planner(llm=AnthropicLLM(api_key=""))

    with pytest.raises(PlannerUnavailableError, match="ANTHROPIC_API_KEY"):
        await planner.decompose("anything")


async def test_no_network_call_is_made_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(PlannerUnavailableError):
        await Planner(llm=AnthropicLLM()).decompose("anything")


# -- rendering -------------------------------------------------------------


def test_describe_lists_every_agent_and_task():
    text = plan(
        researcher={"gather": "Collect sources.", "extract": "Pull claims."},
        writer={"draft": "Write it."},
    ).describe()

    assert "2 agents, 3 tasks" in text
    for fragment in ("researcher", "gather", "extract", "writer", "draft"):
        assert fragment in text
