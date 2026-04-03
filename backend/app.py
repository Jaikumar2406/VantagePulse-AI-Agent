from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from backend.core.lifespan import lifespan
from backend.core.state import server_state
from backend.core.dependencies import _LoadingResponse

# ── Routers ───────────────────────────────────────────────────
from backend.api.routers import health
from backend.api.routers import startup
from backend.api.routers import market
from backend.api.routers import risk
from backend.api.routers import investment
from backend.api.routers import growth
from backend.api.routers import financial
from backend.api.routers import competitor
from backend.api.routers import pipeline


def create_app() -> FastAPI:
    app = FastAPI(
        title="Amukh Capital - Startup Intelligence API",
        version="1.0",
        lifespan=lifespan,
        # Tell FastAPI about our custom security scheme so Swagger renders
        # a single global 'Authorize' button for x-startup-id
    )

    # ── CORS ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Readiness gate — blocks all routes until server is ready ──
    @app.middleware("http")
    async def readiness_gate(request: Request, call_next):
        bypass_paths = {"/health", "/ready"}
        if not server_state.ready and request.url.path not in bypass_paths:
            return JSONResponse(
                status_code=503,
                content={
                    "status":        "starting",
                    "message":       "Server is starting, please wait...",
                    "checks_passed": server_state.checks_passed,
                    "error":         server_state.init_error,
                },
            )
        return await call_next(request)

    # ── All-or-nothing: convert _LoadingResponse to HTTP 202 ──
    @app.exception_handler(_LoadingResponse)
    async def loading_response_handler(request: Request, exc: _LoadingResponse):
        """
        Converts the internal _LoadingResponse sentinel into a proper
        HTTP 202 Accepted response whenever a GET endpoint is called
        before the full pipeline has completed.
        """
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "message": "Full analysis is still running. Please wait...",
            },
        )

    # ── Register Routers ──────────────────────────────────────
    app.include_router(health.router)
    app.include_router(startup.router)
    app.include_router(market.router)
    app.include_router(risk.router)
    app.include_router(investment.router)
    app.include_router(growth.router)
    app.include_router(financial.router)
    app.include_router(competitor.router)
    app.include_router(pipeline.router)

    # ── Custom OpenAPI: inject x-startup-id as global API-key scheme ──
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            routes=app.routes,
        )

        # Define an API-key security scheme so Swagger shows ONE
        # global "Authorize" button where analysts enter the startup ID
        schema.setdefault("components", {}).setdefault("securitySchemes", {})[
            "StartupId"
        ] = {
            "type":        "apiKey",
            "in":          "header",
            "name":        "x-startup-id",
            "description": (
                "Enter the startup_id returned by POST /api/startup/analyze. "
                "This value is sent as the `x-startup-id` header on every request."
            ),
        }

        # Apply the scheme globally to every operation
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation.setdefault("security", [{"StartupId": []}])

        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    return app
