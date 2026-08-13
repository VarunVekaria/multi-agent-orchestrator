"""What happens when things go wrong upstream, hang, or need another try."""

from __future__ import annotations

import asyncio

from multi_agent_orchestrator import EventType, TaskSpec, TaskStatus, workers


async def test_failure_cascades_as_skips(orchestrator):
    orchestrator.submit(TaskSpec("broken", "x", workers.always_fails("boom", duration=0)))
    orchestrator.submit(
        TaskSpec("middle", "y", workers.produces(1), depends_on=["broken"])
    )
    orchestrator.submit(
        TaskSpec("last", "z", workers.produces(2), depends_on=["middle"])
    )

    report = await orchestrator.run()

    assert report.status_of("broken") is TaskStatus.FAILED
    assert "boom" in report.tasks["broken"].error
    # `middle` skips because its prerequisite failed; `last` skips because
    # SKIPPED is itself terminal-and-unsuccessful. Nothing propagates the
    # failure explicitly -- each waiter works it out from a status poll.
    assert report.status_of("middle") is TaskStatus.SKIPPED
    assert report.tasks["middle"].blocked_by == "broken"
    assert report.status_of("last") is TaskStatus.SKIPPED
    assert report.tasks["last"].blocked_by == "middle"
    assert not report.succeeded


async def test_unrelated_branches_are_unaffected_by_a_failure(orchestrator):
    orchestrator.submit(TaskSpec("broken", "x", workers.always_fails(duration=0)))
    orchestrator.submit(
        TaskSpec("downstream", "y", workers.produces(1), depends_on=["broken"])
    )
    orchestrator.submit(TaskSpec("healthy", "z", workers.produces("fine", duration=0.02)))
    orchestrator.submit(
        TaskSpec("also_healthy", "z", workers.produces("ok"), depends_on=["healthy"])
    )

    report = await orchestrator.run()

    assert report.status_of("downstream") is TaskStatus.SKIPPED
    assert report.status_of("healthy") is TaskStatus.COMPLETED
    assert report.status_of("also_healthy") is TaskStatus.COMPLETED
    assert report.result_of("also_healthy") == "ok"


async def test_a_failed_task_never_leaks_a_result(orchestrator):
    async def fails_after_producing(context):
        raise RuntimeError("late failure")

    orchestrator.submit(TaskSpec("f", "x", fails_after_producing))

    report = await orchestrator.run()

    assert report.status_of("f") is TaskStatus.FAILED
    assert report.results == {}


async def test_dependency_timeout_fails_the_waiter(orchestrator):
    orchestrator.submit(TaskSpec("stuck", "x", workers.never_finishes()))
    orchestrator.submit(
        TaskSpec(
            "waiter",
            "y",
            workers.produces(1),
            depends_on=["stuck"],
            dependency_timeout=0.1,
        )
    )

    runner = asyncio.create_task(orchestrator.run())
    # `stuck` never returns, so wait for the waiter to give up, then stop.
    await asyncio.sleep(0.3)
    waiter = orchestrator.tasks["waiter"]

    assert waiter.status is TaskStatus.FAILED
    assert "waited" in (waiter.error or "")
    assert orchestrator.tasks["stuck"].status is TaskStatus.RUNNING

    runner.cancel()
    try:
        await runner
    except asyncio.CancelledError:
        pass


async def test_execution_timeout_fails_the_task(orchestrator):
    orchestrator.submit(TaskSpec("slow", "x", workers.never_finishes(), timeout=0.05))

    report = await orchestrator.run()

    assert report.status_of("slow") is TaskStatus.FAILED
    assert "timed out" in report.tasks["slow"].error


async def test_retries_can_rescue_a_flaky_task(orchestrator):
    orchestrator.submit(
        TaskSpec(
            "flaky",
            "x",
            workers.flaky(succeeds_on_attempt=3, value="recovered", duration=0),
            max_retries=2,
            retry_backoff=0.001,
        )
    )

    report = await orchestrator.run()

    assert report.status_of("flaky") is TaskStatus.COMPLETED
    assert report.result_of("flaky") == "recovered"
    assert report.tasks["flaky"].attempts == 3
    assert len(orchestrator.events.of_type(EventType.TASK_RETRY)) == 2


async def test_retries_are_bounded(orchestrator):
    orchestrator.submit(
        TaskSpec(
            "hopeless",
            "x",
            workers.flaky(succeeds_on_attempt=99, duration=0),
            max_retries=1,
            retry_backoff=0.001,
        )
    )

    report = await orchestrator.run()

    assert report.status_of("hopeless") is TaskStatus.FAILED
    assert report.tasks["hopeless"].attempts == 2


async def test_a_satisfied_dependency_is_never_polled_again(orchestrator):
    """Terminal statuses are cached, so the system does not spin."""
    orchestrator.submit(TaskSpec("quick", "x", workers.produces(1, duration=0)))
    orchestrator.submit(TaskSpec("slow", "y", workers.produces(2, duration=0.2)))
    orchestrator.submit(
        TaskSpec("waiter", "z", workers.produces(3), depends_on=["quick", "slow"])
    )

    await orchestrator.run()
    history = orchestrator.events.history

    satisfied_at = next(
        i
        for i, e in enumerate(history)
        if e.type is EventType.DEPENDENCY_SATISFIED
        and e.detail.get("dependency") == "quick"
    )
    later_polls_for_quick = [
        e
        for e in history[satisfied_at + 1 :]
        if e.type is EventType.POLL_ISSUED and e.detail.get("dependency") == "quick"
    ]

    assert later_polls_for_quick == []
    # The slow dependency, meanwhile, was polled repeatedly -- proving the
    # waiter really was still looping and simply skipping the settled one.
    polls_for_slow = [
        e
        for e in history
        if e.type is EventType.POLL_ISSUED and e.detail.get("dependency") == "slow"
    ]
    assert len(polls_for_slow) > 1


async def test_backoff_keeps_poll_counts_sane(orchestrator):
    """A long wait must not mean an unbounded number of polls."""
    orchestrator.submit(TaskSpec("slow", "x", workers.produces(1, duration=0.4)))
    orchestrator.submit(
        TaskSpec("waiter", "y", workers.produces(2), depends_on=["slow"])
    )

    await orchestrator.run()

    polls = orchestrator.events.of_type(EventType.POLL_ISSUED)
    # With the test policy capped at 20ms, a 400ms wait is ~20-40 polls. The
    # bound here is loose on purpose; it only needs to catch a busy-loop.
    assert len(polls) < 100
