# multi-agent-orchestrator

Multiple independent agents, each owning a set of tasks. A task is either
independent or depends on other tasks. An agent starts its independent tasks
immediately; for a dependent task it **polls the agent that owns each
prerequisite** until that agent reports the task complete.

There is no central scheduler. Nothing computes a plan and hands out work.
Agents discover each other through a registry, ask each other questions, and
the execution order falls out of the dependencies on its own.

```bash
pip install -e ".[dev,demo]"
python -m demo.demo_pipeline
```

## The idea in 20 lines

```python
from multi_agent_orchestrator import Orchestrator, TaskSpec, workers

orch = Orchestrator()

# agent "fetcher" owns two tasks with no prerequisites -> both start at once
orch.submit(TaskSpec("fetch_users", "fetcher", workers.produces(["ada", "grace"])))
orch.submit(TaskSpec("fetch_orders", "fetcher", workers.produces([("ada", "keyboard")])))

# agent "processor" owns a task that waits on both, across an agent boundary
orch.submit(TaskSpec(
    "join_data", "processor",
    workers.combines(lambda deps: (deps["fetch_users"], deps["fetch_orders"])),
    depends_on=["fetch_users", "fetch_orders"],
))

report = orch.run_sync()
print(report.summary())          # 3 tasks in 0.31s: 3 completed
print(report.result_of("join_data"))
```

A task names its prerequisites by id only. It does not need to know which agent
produces them — the registry resolves the owner at poll time.

## How coordination actually works

Every agent exposes exactly one thing to its peers:

```python
agent.get_task_status(task_id) -> TaskStatusResponse   # frozen: id, status, result, error
```

That is the entire inter-agent protocol. A task with prerequisites loops:

1. For each unsettled prerequisite, resolve its owner and ask that owner for status.
2. `COMPLETED` → record the result and **stop polling that one for good**.
3. `FAILED` or `SKIPPED` → stop waiting; this task becomes `SKIPPED`.
4. Still running → keep it in the pending set and try again after a backoff.

Four properties make this workable rather than naive:

- **Settled prerequisites are cached.** Each round only polls what is still
  unresolved, so poll traffic falls as the graph resolves.
- **Exponential backoff with jitter**, capped. Without it, cost scales with
  graph size instead of with work done, and agents started together drift into
  synchronised poll bursts.
- **Failure propagates through the protocol itself.** A waiter that sees a
  failed prerequisite marks itself `SKIPPED`, which is terminal-and-unsuccessful,
  so *its* dependents skip on their next poll. There is no separate propagation
  pass.
- **Waiting is bounded** by `dependency_timeout`, so a stuck upstream degrades
  into a failure instead of a hung process.

Per-task coroutines mean a blocked task never holds up a ready sibling on the
same agent, and `max_concurrency` bounds how many of an agent's tasks execute
at once.

## Plans that could never finish are rejected up front

Decentralised polling has one sharp edge: waiting on something that will never
settle is indistinguishable from waiting on something slow. So `validate()`
runs before any agent starts and refuses:

| Fault | Why it is fatal | Error |
| --- | --- | --- |
| Dependency cycle | every agent in the cycle politely polls the next, forever | `DependencyCycleError` |
| Dependency no agent owns | the poll never gets a terminal answer | `UnknownDependencyError` |
| Duplicate task id | task ids are the routing key for status lookups | `DuplicateTaskError` |

`Orchestrator.run()` calls `validate()` itself, so a bad plan raises instead of
hanging.

## Task options

```python
TaskSpec(
    id="report", agent_id="reporter", handler=my_handler,
    depends_on=["enrich"],
    timeout=30.0,             # budget for one execution attempt
    dependency_timeout=120.0, # budget for waiting on prerequisites
    max_retries=2,            # retry the handler on failure
    retry_backoff=0.5,
)
```

A handler receives an `ExecutionContext` with `task_id`, `agent_id`, `attempt`,
and `deps` — the prerequisites' results keyed by task id. Handlers may be async
or sync; sync handlers block the event loop and cannot be interrupted by
`timeout`, so anything slow should be async.

## Statuses

`PENDING → WAITING → READY → RUNNING → {COMPLETED, FAILED, SKIPPED}`

`SKIPPED` specifically means a prerequisite did not succeed; `Task.blocked_by`
names which one.

## Observability

Agents emit structured events (`poll_issued`, `dependency_satisfied`,
`task_started`, …) to an `EventBus`. The demo renders them as a live table and
a timeline; the test suite asserts ordering off the same log, which is why the
tests do not depend on wall-clock timing.

## The demo

```bash
python -m demo.demo_pipeline                          # happy path
python -m demo.demo_pipeline --fail-task fetch_orders # SKIPPED cascade
python -m demo.demo_pipeline --no-live --speed 0.5    # plain output, faster
```

Three agents. `fetcher` has two independent tasks; `processor` joins them and
then enriches its own output (a same-agent dependency, which takes the exact
same polling path); `reporter` has one independent task and one that waits on
the whole chain. The timeline shows the independent tasks running concurrently
while the dependent ones wait:

```
timeline (. waiting, # running)
 fetch_users  | #########
 fetch_orders | #############
 join_data    | ...................######
 enrich       | ...........................######
 warmup       | #################
 report       | ................................................#####
```

## Layout

| File | Role |
| --- | --- |
| `models.py` | `TaskStatus`, `TaskSpec`, `Task`, `TaskStatusResponse`, `ExecutionContext` |
| `registry.py` | agent discovery (`owner_of`) and graph validation |
| `transport.py` | `StatusClient` protocol + in-process implementation |
| `agent.py` | the agent: serving status, the polling waiter, execution |
| `orchestrator.py` | assembly, validation, lifecycle, `RunReport` |
| `events.py` | structured event log |
| `workers.py` | demo handlers |

## Running out of process

Agents currently run as asyncio peers in one process. The polling code talks
only to the `StatusClient` protocol in `transport.py` and never learns where
the agent it questions actually lives, so putting agents behind HTTP means
adding a second implementation of that one interface (plus a registry lookup
that crosses the network). No other module changes.

## Tests

```bash
python -m pytest -q
```

Covers validation, waiting and polling behaviour, cross-agent ordering,
concurrency limits, the failure cascade, both timeouts, retries, and the
poll-caching guarantee.
