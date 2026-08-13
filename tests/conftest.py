"""Shared fixtures.

Tests use a deterministic poll policy: no jitter and a tiny fixed interval, so
runs are fast and repeatable. Ordering assertions read off the event log rather
than wall-clock timings, so a loaded machine cannot make them flap.
"""

from __future__ import annotations

import random

import pytest

from multi_agent_orchestrator import Orchestrator, PollPolicy

FAST_POLL = PollPolicy(base_interval=0.005, max_interval=0.02, multiplier=1.2, jitter=0.0)


@pytest.fixture
def orchestrator() -> Orchestrator:
    return Orchestrator(poll_policy=FAST_POLL, rng=random.Random(0))
