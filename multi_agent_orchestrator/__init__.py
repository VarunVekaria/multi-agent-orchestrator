"""Multi-agent task orchestrator.

Agents own tasks. Independent tasks start at once; a dependent task polls the
agent that owns each prerequisite until it reports success. There is no central
scheduler -- execution order emerges from the dependencies themselves.

    from multi_agent_orchestrator import Orchestrator, TaskSpec, workers

    orch = Orchestrator()
    orch.submit(TaskSpec("fetch", "fetcher", workers.produces([1, 2, 3])))
    orch.submit(TaskSpec("sum", "reducer", workers.combines(
        lambda deps: sum(deps["fetch"])), depends_on=["fetch"]))
    report = orch.run_sync()
"""

from . import workers
from .agent import Agent, PollPolicy
from .errors import (
    DependencyCycleError,
    DependencyFailedError,
    DependencyTimeoutError,
    DuplicateTaskError,
    OrchestratorError,
    UnknownAgentError,
    UnknownDependencyError,
    UnknownTaskError,
    ValidationError,
)
from .events import Event, EventBus, EventType
from .models import (
    ExecutionContext,
    Handler,
    Task,
    TaskSpec,
    TaskStatus,
    TaskStatusResponse,
)
from .orchestrator import Orchestrator, RunReport, build
from .registry import AgentRegistry
from .transport import LocalStatusClient, StatusClient

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentRegistry",
    "DependencyCycleError",
    "DependencyFailedError",
    "DependencyTimeoutError",
    "DuplicateTaskError",
    "Event",
    "EventBus",
    "EventType",
    "ExecutionContext",
    "Handler",
    "LocalStatusClient",
    "Orchestrator",
    "OrchestratorError",
    "PollPolicy",
    "RunReport",
    "StatusClient",
    "Task",
    "TaskSpec",
    "TaskStatus",
    "TaskStatusResponse",
    "UnknownAgentError",
    "UnknownDependencyError",
    "UnknownTaskError",
    "ValidationError",
    "build",
    "workers",
]
