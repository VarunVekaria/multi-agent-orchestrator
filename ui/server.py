"""The front door: accept a query, acknowledge it, do nothing with it.

This is deliberately inert. ``POST /api/query`` records the query and returns
an id -- it does not plan, dispatch, or execute anything, and this module does
not import :mod:`multi_agent_orchestrator` at all. The endpoint exists so the
request path is real and there is exactly one obvious place for the planner to
attach later, rather than a UI that has to grow a backend when that happens.
"""

from __future__ import annotations

import logging
from itertools import count
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

STATIC_DIR = Path(__file__).parent / "static"

# Configured at import, not in main(), because the usual way to run this is
# `python -m uvicorn ui.server:app` -- which never calls main(). uvicorn
# configures its own loggers and leaves the root one alone, so without this the
# query log line below is silently dropped. basicConfig is a no-op when the root
# logger already has handlers, so this defers to any caller that set up logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="multi-agent-orchestrator", docs_url=None, redoc_url=None)

#: Monotonic ids for accepted queries. In-memory and reset on restart, which is
#: fine while nothing depends on them surviving the process.
_query_ids: Iterator[int] = count(1)


class QueryRequest(BaseModel):
    """A submitted query.

    Validating here rather than trusting the client keeps the API honest on its
    own terms -- the page disables its send button on empty input, but that is a
    convenience, not a guarantee.
    """

    query: str = Field(min_length=1, max_length=10_000)

    @field_validator("query")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must contain non-whitespace characters")
        return stripped


class QueryAccepted(BaseModel):
    id: int
    received: bool = True


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/query")
async def submit_query(request: QueryRequest) -> QueryAccepted:
    """Take the query and stop.

    The log line is the only observable effect, and that is the point: it proves
    the round trip works before there is anything on the other end of it.
    """
    query_id = next(_query_ids)
    logger.info("query %d received: %s", query_id, request.query)
    return QueryAccepted(id=query_id)


# Mounted last so it cannot shadow the routes declared above.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    """Run the dev server. Entry point for ``python -m ui.server``."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
