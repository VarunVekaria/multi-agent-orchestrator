"""Agent-level behaviour: when work starts, and what one agent shows another."""

from __future__ import annotations

import asyncio

import pytest

from multi_agent_orchestrator import (
    Agent,
    EventType,
    ExecutionContext,
    TaskSpec,
    TaskStatus,
    UnknownTaskError,
    workers,
)


async def test_independent_tasks_never_wait_and_never_poll(orchestrator):
    orchestrator.submit(TaskSpec("solo", "x", workers.produces(7, duration=0)))

    report = await orchestrator.run()

    assert report.status_of("solo") is TaskStatus.COMPLETED
    assert report.result_of("solo") == 7
    assert orchestrator.events.for_task("solo") != []
    assert orchestrator.events.of_type(EventType.TASK_WAITING) == []
    assert orchestrator.events.of_type(EventType.POLL_ISSUED) == []


async def test_dependent_task_waits_polls_then_runs(orchestrator):
    orchestrator.submit(TaskSpec("first", "x", workers.produces("A", duration=0.05)))
    orchestrator.submit(
        TaskSpec("second", "y", workers.produces("B"), depends_on=["first"])
    )

    await orchestrator.run()
    events = orchestrator.events

    assert events.index_of(EventType.TASK_WAITING, "second") >= 0
    assert events.of_type(EventType.POLL_ISSUED) != []
    # The wait has to end before the work begins, and only after the
    # prerequisite actually reported success.
    assert (
        events.index_of(EventType.TASK_WAITING, "second")
        < events.index_of(EventType.TASK_COMPLETED, "first")
        < events.index_of(EventType.TASK_STARTED, "second")
    )


async def test_dependency_results_reach_the_handler(orchestrator):
    seen: dict[str, object] = {}

    async def capture(context: ExecutionContext):
        seen.update(context.deps)
        return sum(context.deps.values())

    orchestrator.submit(TaskSpec("a", "x", workers.produces(2, duration=0)))
    orchestrator.submit(TaskSpec("b", "y", workers.produces(3, duration=0.02)))
    orchestrator.submit(TaskSpec("c", "z", capture, depends_on=["a", "b"]))

    report = await orchestrator.run()

    assert seen == {"a": 2, "b": 3}
    assert report.result_of("c") == 5


async def test_same_agent_dependency_uses_the_same_polling_path(orchestrator):
    """A task waiting on its own agent's task gets no special treatment."""
    orchestrator.submit(TaskSpec("up", "solo", workers.produces(1, duration=0.03)))
    orchestrator.submit(
        TaskSpec("down", "solo", workers.produces(2), depends_on=["up"])
    )

    report = await orchestrator.run()

    polls = orchestrator.events.of_type(EventType.POLL_ISSUED)
    assert polls, "a same-agent dependency should still be polled"
    assert all(p.detail["owner"] == "solo" for p in polls)
    assert report.succeeded


async def test_status_snapshot_hides_results_until_completion():
    agent = Agent("x")
    agent.task("t", workers.produces("value", duration=0))

    before = agent.get_task_status("t")
    assert before.status is TaskStatus.PENDING
    assert before.result is None

    await agent_run(agent)

    after = agent.get_task_status("t")
    assert after.status is TaskStatus.COMPLETED
    assert after.result == "value"


async def agent_run(agent: Agent):
    """Run a lone agent, wiring the registry it would normally get."""
    from multi_agent_orchestrator import AgentRegistry, LocalStatusClient

    registry = AgentRegistry()
    registry.register(agent)
    agent.attach(registry=registry, status_client=LocalStatusClient(registry))
    return await agent.run()


async def test_asking_about_a_foreign_task_is_an_error():
    agent = Agent("x")
    agent.task("mine", workers.produces(1))

    with pytest.raises(UnknownTaskError):
        agent.get_task_status("not_mine")


async def test_max_concurrency_is_respected(orchestrator):
    live = 0
    peak = 0

    async def tracked(context: ExecutionContext):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.05)
        live -= 1
        return context.task_id

    agent = Agent("limited", max_concurrency=2)
    orchestrator.add_agent(agent)
    for i in range(6):
        agent.task(f"t{i}", tracked)

    report = await orchestrator.run()

    assert report.succeeded
    assert peak == 2


async def test_sync_handlers_are_supported(orchestrator):
    orchestrator.submit(TaskSpec("s", "x", lambda context: context.task_id.upper()))

    report = await orchestrator.run()

    assert report.result_of("s") == "S"
