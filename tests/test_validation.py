"""Plans that could never finish must be rejected before anything starts.

This is the price of decentralised coordination: an agent waiting on something
that will never settle just waits. Every such shape is a validation error.
"""

from __future__ import annotations

import pytest

from multi_agent_orchestrator import (
    Agent,
    DependencyCycleError,
    DuplicateTaskError,
    EventType,
    TaskSpec,
    UnknownDependencyError,
    workers,
)


def spec(id: str, agent_id: str, *deps: str) -> TaskSpec:
    return TaskSpec(id, agent_id, workers.produces(id, duration=0), depends_on=deps)


def test_two_task_cycle_is_rejected(orchestrator):
    orchestrator.submit(spec("a", "x", "b"))
    orchestrator.submit(spec("b", "y", "a"))

    with pytest.raises(DependencyCycleError) as exc:
        orchestrator.validate()

    cycle = set(exc.value.cycles[0])
    assert cycle == {"a", "b"}


def test_longer_cycle_is_rejected(orchestrator):
    orchestrator.submit(spec("a", "x", "c"))
    orchestrator.submit(spec("b", "y", "a"))
    orchestrator.submit(spec("c", "z", "b"))

    with pytest.raises(DependencyCycleError) as exc:
        orchestrator.validate()

    assert set(exc.value.cycles[0]) == {"a", "b", "c"}


async def test_cycle_is_caught_before_any_agent_runs(orchestrator):
    orchestrator.submit(spec("a", "x", "b"))
    orchestrator.submit(spec("b", "y", "a"))

    with pytest.raises(DependencyCycleError):
        await orchestrator.run()

    # Nothing was allowed to start, so nothing is left half-run.
    assert orchestrator.events.of_type(EventType.TASK_STARTED) == []
    assert orchestrator.events.of_type(EventType.AGENT_STARTED) == []


def test_unknown_dependency_is_rejected(orchestrator):
    orchestrator.submit(spec("a", "x", "does_not_exist"))

    with pytest.raises(UnknownDependencyError) as exc:
        orchestrator.validate()

    assert exc.value.task_id == "a"
    assert exc.value.missing == "does_not_exist"


def test_duplicate_task_id_across_agents_is_rejected(orchestrator):
    orchestrator.submit(spec("shared", "x"))

    with pytest.raises(DuplicateTaskError) as exc:
        orchestrator.submit(spec("shared", "y"))

    assert exc.value.task_id == "shared"
    assert {exc.value.first_agent, exc.value.second_agent} == {"x", "y"}


def test_self_dependency_is_rejected_at_spec_construction():
    with pytest.raises(ValueError, match="depends on itself"):
        spec("a", "x", "a")


def test_task_assigned_to_the_wrong_agent_is_rejected():
    agent = Agent("x")
    with pytest.raises(ValueError, match="assigned to agent"):
        agent.add_task(spec("a", "other"))


def test_depends_on_is_snapshotted_at_construction():
    deps = ["a"]
    s = TaskSpec("b", "x", workers.produces(1), depends_on=deps)
    deps.append("sneaky")
    # Mutating the caller's list after the fact must not change a graph that
    # has already been validated.
    assert s.depends_on == ("a",)


def test_execution_layers_describe_the_graph(orchestrator):
    orchestrator.submit(spec("a", "x"))
    orchestrator.submit(spec("b", "x"))
    orchestrator.submit(spec("c", "y", "a", "b"))
    orchestrator.submit(spec("d", "z", "c"))

    assert orchestrator.execution_layers() == [["a", "b"], ["c"], ["d"]]
