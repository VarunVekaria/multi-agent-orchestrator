"""A three-agent pipeline you can watch run.

    python -m demo.demo_pipeline
    python -m demo.demo_pipeline --fail-task fetch_orders

The graph is arranged so each branch of the coordination logic is visible:

* ``fetcher`` holds two independent tasks -- they start instantly, together.
* ``processor.join_data`` waits on both of them, across agent boundaries.
* ``processor.enrich`` waits on a task its *own* agent owns, which takes the
  identical polling path -- there is no shortcut for same-agent dependencies.
* ``reporter.warmup`` is independent and should be visibly RUNNING while
  ``reporter.report`` is still WAITING, proving a blocked task does not hold up
  a ready sibling.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from multi_agent_orchestrator import (
    EventType,
    Orchestrator,
    TaskSpec,
    TaskStatus,
    workers,
)

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)

STATUS_STYLE = {
    TaskStatus.PENDING: "dim",
    TaskStatus.WAITING: "yellow",
    TaskStatus.READY: "cyan",
    TaskStatus.RUNNING: "bold blue",
    TaskStatus.COMPLETED: "bold green",
    TaskStatus.FAILED: "bold red",
    TaskStatus.SKIPPED: "magenta",
}


# -- the pipeline ---------------------------------------------------------


def _join(deps: Mapping[str, Any]) -> dict[str, Any]:
    users = deps["fetch_users"]
    orders = deps["fetch_orders"]
    by_user: dict[str, list[str]] = {u: [] for u in users}
    for user, item in orders:
        by_user.setdefault(user, []).append(item)
    return by_user


def build_pipeline(fail_task: str | None, speed: float) -> Orchestrator:
    def secs(base: float) -> float:
        return base * speed

    orchestrator = Orchestrator()
    specs = [
        TaskSpec(
            "fetch_users",
            "fetcher",
            workers.produces(["ada", "grace", "alan"], duration=secs(0.6)),
        ),
        TaskSpec(
            "fetch_orders",
            "fetcher",
            workers.produces(
                [("ada", "keyboard"), ("grace", "compiler"), ("ada", "monitor")],
                duration=secs(0.9),
            ),
        ),
        TaskSpec(
            "join_data",
            "processor",
            workers.combines(_join, duration=secs(0.4)),
            depends_on=["fetch_users", "fetch_orders"],
        ),
        TaskSpec(
            "enrich",
            "processor",
            workers.combines(
                lambda deps: {
                    user: {"items": items, "count": len(items)}
                    for user, items in deps["join_data"].items()
                },
                duration=secs(0.4),
            ),
            depends_on=["join_data"],
        ),
        TaskSpec(
            "warmup",
            "reporter",
            workers.produces("cache warm", duration=secs(1.2)),
        ),
        TaskSpec(
            "report",
            "reporter",
            workers.combines(
                lambda deps: "\n".join(
                    f"  {user}: {info['count']} item(s) -> {', '.join(info['items']) or '-'}"
                    for user, info in sorted(deps["enrich"].items())
                ),
                duration=secs(0.3),
            ),
            depends_on=["enrich"],
        ),
    ]

    if fail_task is not None:
        ids = {spec.id for spec in specs}
        if fail_task not in ids:
            raise typer.BadParameter(
                f"unknown task {fail_task!r}; choose one of {', '.join(sorted(ids))}"
            )
        specs = [
            TaskSpec(
                s.id,
                s.agent_id,
                workers.always_fails("forced failure for the demo"),
                depends_on=s.depends_on,
            )
            if s.id == fail_task
            else s
            for s in specs
        ]

    orchestrator.submit_all(specs)
    orchestrator.validate()
    return orchestrator


# -- rendering ------------------------------------------------------------


def render_table(orchestrator: Orchestrator) -> Table:
    table = Table(title="agents & tasks", title_style="bold", expand=False)
    table.add_column("agent", style="cyan", no_wrap=True)
    table.add_column("task", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("waits on", style="dim", no_wrap=True)
    table.add_column("detail", no_wrap=True, overflow="ellipsis")

    for agent in orchestrator.registry.agents:
        for task in agent.tasks:
            if task.status is TaskStatus.SKIPPED:
                detail = f"blocked by {task.blocked_by}"
            elif task.error:
                detail = task.error
            elif task.status.is_successful and task.duration is not None:
                detail = f"{task.duration:.2f}s"
            else:
                detail = ""
            table.add_row(
                agent.id,
                task.id,
                Text(str(task.status), style=STATUS_STYLE[task.status]),
                ", ".join(task.spec.depends_on) or "-",
                detail,
            )
    return table


def _glyphs() -> tuple[str, str]:
    """Bar characters the current console can actually encode.

    Windows terminals still default to a legacy codepage, which cannot encode
    the block glyphs; fall back to ASCII rather than crashing on output.
    """
    encoding = getattr(console.file, "encoding", None) or "ascii"
    try:
        "·█".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return ".", "#"
    return "·", "█"


def render_timeline(orchestrator: Orchestrator, width: int = 54) -> Table:
    """Reconstruct who was waiting and who was working, from the event log."""
    history = orchestrator.events.history
    if not history:
        return Table()
    origin = history[0].at
    span = max(e.at for e in history) - origin or 1e-9

    terminal = {
        EventType.TASK_COMPLETED,
        EventType.TASK_FAILED,
        EventType.TASK_SKIPPED,
    }
    marks: dict[str, dict[str, float]] = {}
    for event in history:
        if event.task_id is None:
            continue
        m = marks.setdefault(event.task_id, {})
        if event.type is EventType.TASK_SUBMITTED:
            m.setdefault("submitted", event.at)
        elif event.type is EventType.TASK_WAITING:
            m.setdefault("wait", event.at)
        elif event.type is EventType.TASK_STARTED:
            m.setdefault("run", event.at)
        elif event.type in terminal:
            m["end"] = event.at

    waiting_glyph, running_glyph = _glyphs()
    table = Table(
        title=f"timeline ({waiting_glyph} waiting, {running_glyph} running)",
        title_style="bold",
    )
    table.add_column("task", no_wrap=True)
    table.add_column("", no_wrap=True)

    def col(at: float) -> int:
        return int((at - origin) / span * (width - 1))

    for task_id, m in marks.items():
        end = col(m.get("end", origin + span))
        run = col(m["run"]) if "run" in m else end
        wait = col(m.get("wait", m.get("submitted", origin)))
        bar = Text()
        bar.append(" " * wait)
        bar.append(waiting_glyph * max(0, run - wait), style="yellow")
        bar.append(running_glyph * max(1, end - run), style="blue")
        table.add_row(task_id, bar)
    return table


async def _run_live(orchestrator: Orchestrator):
    from rich.live import Live

    with Live(render_table(orchestrator), console=console, refresh_per_second=20) as live:
        runner = asyncio.create_task(orchestrator.run())
        while not runner.done():
            live.update(render_table(orchestrator))
            await asyncio.sleep(0.05)
        live.update(render_table(orchestrator))
        return await runner


# -- cli ------------------------------------------------------------------


@app.command()
def main(
    fail_task: str = typer.Option(
        None, "--fail-task", help="Force this task to raise, to show the SKIPPED cascade."
    ),
    speed: float = typer.Option(
        1.0, "--speed", help="Multiplier on every worker's duration."
    ),
    live: bool = typer.Option(True, "--live/--no-live", help="Animate the status table."),
) -> None:
    orchestrator = build_pipeline(fail_task, speed)

    console.print("[bold]dependency layers[/bold] (descriptive only -- nothing reads this at runtime)")
    for i, layer in enumerate(orchestrator.execution_layers()):
        console.print(f"  {i}: {', '.join(layer)}")
    console.print()

    if live:
        report = asyncio.run(_run_live(orchestrator))
    else:
        report = orchestrator.run_sync()
        console.print(render_table(orchestrator))

    console.print()
    console.print(render_timeline(orchestrator))
    console.print()

    polls = len(orchestrator.events.of_type(EventType.POLL_ISSUED))
    console.print(f"[bold]{report.summary()}[/bold]  ({polls} status polls issued)")

    if report.status_of("report") is TaskStatus.COMPLETED:
        console.print("\n[bold]report[/bold]")
        console.print(report.result_of("report"))
    else:
        for task in report.failed + report.skipped:
            reason = (
                f"blocked by {task.blocked_by}" if task.blocked_by else task.error
            )
            console.print(f"[red]{task.id}[/red]: {task.status} -- {reason}")

    raise typer.Exit(0 if report.succeeded else 1)


if __name__ == "__main__":
    app()
