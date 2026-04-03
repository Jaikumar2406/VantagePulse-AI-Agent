import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.core.state import server_state

router = APIRouter(tags=["Operations"])


@router.get("/health")
def health_check():
    """Always reachable. Returns server and readiness status."""
    uptime_seconds = round(time.time() - server_state.started_at, 1)
    return {
        "status":         "healthy" if server_state.ready else "starting",
        "ready":          server_state.ready,
        "uptime_seconds": uptime_seconds,
        "checks_passed":  server_state.checks_passed,
        "error":          server_state.init_error,
    }


@router.get("/ready")
def readiness_probe():
    """
    Kubernetes / load-balancer style readiness probe.
    Returns 200 when ready, 503 when not.
    """
    if server_state.ready:
        return {"status": "ready"}
    return JSONResponse(
        status_code=503,
        content={
            "status":  "not_ready",
            "message": server_state.init_error or "Server is initializing.",
            "checks_passed": server_state.checks_passed,
        },
    )
