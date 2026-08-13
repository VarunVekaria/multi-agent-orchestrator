"""Whole-system behaviour across several agents."""

from __future__ import annotations

import asyncio

from multi_agent_orchestrator import (
    Agent,
    EventType,
    ExecutionContext,
    TaskSpec,
    TaskStatus,
    workers,
)


async def test_cross_agent_chain_runs_in_dependency_order(orchestrator):
    orchestrator.submit(TaskSpec("fetch", "fetcher", workers.produces([1, 2, 3], duration=0.02)))
    orchestrator.submit(
        TaskSpec(
            "double",
            "processor",
            workers.combines(lambda deps: [n * 2 for n in deps["fetch"]], duration=0.02),
            depends_on=["fetch"],
        )
    )
    orchestrator.submit(
        TaskSpec(
            "total",
            "reporter",
            workers.combines(lambda deps: sum(deps["double"])),
            depends_on=["double"],
        )
    )

    report = await orchestrator.run()
    events = orchestrator.events

    assert report.succeeded
    assert report.result_of("total") == 12
    assert (
        events.index_of(EventType.TASK_COMPLETED, "fetch")
        < events.index_of(EventType.TASK_STARTED, "double")
        < events.index_of(EventType.TASK_COMPLETED, "double")
        < events.index_of(EventType.TASK_STARTED, "total")
    )


async def test_a_blocked_task_does_not_hold_up_a_ready_sibling(orchestrator):
    """The reporter's independent task must run while its other task waits."""
    orchestrator.submit(TaskSpec("slow_upstream", "fetcher", workers.produces("u", duration=0.15)))
    orchestrator.submit(TaskSpec("warmup", "reporter", workers.produces("w", duration=0.01)))
    orchestrator.submit(
        TaskSpec(
            "report",
            "reporter",
            workers.produces("r"),
            depends_on=["slow_upstream"],
        )
    )

    await orchestrator.run()
    events = orchestrator.events

    # warmup finished long before its sibling was even unblocked.
    assert (
        events.index_of(EventType.TASK_COMPLETED, "warmup")
        < events.index_of(EventType.TASK_COMPLETED, "slow_upstream")
        < events.index_of(EventType.TASK_STARTED, "report")
    )


async def test_diamond_graph_fans_out_and_back_in(orchestrator):
    orchestrator.submit(TaskSpec("root", "a", workers.produces(10, duration=0.01)))
    orchestrator.submit(
        TaskSpec(
            "left",
            "b",
            workers.combines(lambda d: d["root"] + 1, duration=0.05),
            depends_on=["root"],
        )
    )
    orchestrator.submit(
        TaskSpec(
            "right",
            "c",
            workers.combines(lambda d: d["root"] + 2, duration=0.01),
            depends_on=["root"],
        )
    )
    orchestrator.submit(
        TaskSpec(
            "merge",
            "d",
            workers.combines(lambda d: d["left"] * d["right"]),
            depends_on=["left", "right"],
        )
    )

    report = await orchestrator.run()

    assert report.result_of("merge") == 11 * 12
    events = orchestrator.events
    # The fan-out branches overlap: the faster one finishes before the slower
    # one does, and merge waits for both.
    assert (
        events.index_of(EventType.TASK_COMPLETED, "right")
        < events.index_of(EventType.TASK_COMPLETED, "left")
        < events.index_of(EventType.TASK_STARTED, "merge")
    )


async def test_independent_agents_run_concurrently(orchestrator):
    """Three agents with unrelated work should overlap, not queue up."""
    running: set[str] = set()
    overlap = 0

    async def tracked(context: ExecutionContext):
        nonlocal overlap
        running.add(context.task_id)
        await asyncio.sleep(0.05)
        overlap = max(overlap, len(running))
        running.discard(context.task_id)
        return context.task_id

    for agent_id in ("a", "b", "c"):
        orchestrator.submit(TaskSpec(f"task_{agent_id}", agent_id, tracked))

    report = await orchestrator.run()

    assert report.succeeded
    assert overlap == 3


async def test_report_aggregates_results_and_summary(orchestrator):
    orchestrator.submit(TaskSpec("a", "x", workers.produces(1, duration=0)))
    orchestrator.submit(TaskSpec("b", "y", workers.produces(2, duration=0), depends_on=["a"]))

    report = await orchestrator.run()

    assert report.results == {"a": 1, "b": 2}
    assert report.succeeded
    assert "2 tasks" in report.summary()
    assert "2 completed" in report.summary()


async def test_agents_can_be_built_directly_and_added(orchestrator):
    agent = Agent("manual")
    agent.task("one", workers.produces("first", duration=0))
    agent.task("two", workers.produces("second"), depends_on=["one"])
    orchestrator.add_agent(agent)

    report = await orchestrator.run()

    assert report.succeeded
    assert agent.results == {"one": "first", "two": "second"}
