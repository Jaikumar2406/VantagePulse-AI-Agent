"""
Central dependency module.

Two exports used by every router:
  • get_startup_id  — reads `x-startup-id` from the request header
  • get_pipeline_result — validates existence + global all-or-nothing pipeline gate
"""

from fastapi import Header, HTTPException
from fastapi.responses import JSONResponse

from backend.core.database import db


# ─────────────────────────────────────────────────────────────
# 1. HEADER-BASED STARTUP ID
# ─────────────────────────────────────────────────────────────

def get_startup_id(
    x_startup_id: str = Header(
        alias="x-startup-id",
        description="Unique startup analysis ID returned by POST /api/startup/analyze",
    )
) -> str:
    """
    FastAPI dependency — extracts `x-startup-id` from the request header.
    FastAPI documents this in Swagger as a header parameter for every endpoint
    that depends on it. The OpenAPI security scheme in app.py exposes a single
    global 'Authorize' button so the value only needs to be entered once.
    """
    return x_startup_id


# ─────────────────────────────────────────────────────────────
# 2. ALL-OR-NOTHING PIPELINE GATE
# ─────────────────────────────────────────────────────────────

def get_pipeline_result(startup_id: str):
    """
    Shared validator used by every GET endpoint to enforce the STRICT all-or-nothing rule.

    Returns
    -------
    dict
        The fully processed result dict when global_status == "completed".

    Raises
    ------
    HTTPException 404   — unknown startup_id
    HTTPException 500   — pipeline failed anywhere
    JSONResponse  202   — pipeline still in progress (short-circuits via _LoadingResponse)

    NOTE: When the pipeline is not yet "completed", this function raises a
    JSONResponse (which FastAPI propagates as the actual HTTP response), so
    callers never need to check for None — they either get the full result dict
    or the response is short-circuited entirely.
    """
    if startup_id not in db:
        raise HTTPException(status_code=404, detail="Startup ID not found")

    record = db[startup_id]
    global_status = record.get("global_status", "not_started")

    # ── Pipeline failed ────────────────────────────────────────
    if global_status == "failed":
        raise HTTPException(
            status_code=500,
            detail={
                "status":  "failed",
                "message": "Pipeline execution failed.",
                "error":   record.get("global_error", "Unknown error"),
            },
        )

    # ── Pipeline not yet complete → 202 loading gate ──────────
    if global_status != "completed":
        raise _LoadingResponse(startup_id, global_status)

    # ── Completed — return raw result dict ────────────────────
    return record.get("result", {})


class _LoadingResponse(Exception):
    """
    Internal sentinel raised to short-circuit a route handler.
    Converted to a proper 202 JSONResponse by the exception handler
    registered in app.py.
    """

    def __init__(self, startup_id: str, global_status: str):
        self.startup_id    = startup_id
        self.global_status = global_status
