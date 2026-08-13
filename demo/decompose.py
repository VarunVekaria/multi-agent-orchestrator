"""Run the decomposer against a query and print what it produced.

    python -m demo.decompose "Research 3 Python web frameworks and compare them"

Needs ANTHROPIC_API_KEY. Pass --demo to see the output shape using a canned
plan instead, with no key and no network.
"""

from __future__ import annotations

import sys

from multi_agent_orchestrator.planning import (
    DecomposedPlan,
    Planner,
    PlanningError,
    ScriptedLLM,
)

CANNED = DecomposedPlan(
    agents=[
        {
            "id": "researcher",
            "role": "Gathers source material and pulls out the facts that matter.",
            "tasks": [
                {
                    "id": "collect_sources",
                    "description": "Find primary sources covering each option.",
                },
                {
                    "id": "extract_findings",
                    "description": "Pull the comparable facts out of each source.",
                },
            ],
        },
        {
            "id": "analyst",
            "role": "Turns raw findings into a defensible comparison.",
            "tasks": [
                {
                    "id": "build_comparison",
                    "description": "Score each option on the agreed criteria.",
                }
            ],
        },
        {
            "id": "writer",
            "role": "Produces the finished write-up.",
            "tasks": [
                {
                    "id": "draft_report",
                    "description": "Write the comparison as a short report.",
                }
            ],
        },
    ]
)


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--demo"]
    offline = "--demo" in argv

    if not args:
        print(__doc__)
        return 2

    query = " ".join(args)
    planner = Planner(llm=ScriptedLLM([CANNED])) if offline else Planner()

    try:
        plan = planner.decompose_sync(query)
    except PlanningError as error:
        # These are the caller's to act on (missing key, refusal), so print the
        # message rather than a traceback.
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"query: {query}\n")
    print(plan.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
