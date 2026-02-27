"""
FastAPI Application Factory.

Configures the API server with:
- CORS middleware
- Rate limiting
- Structured error handling
- Health checks
- OpenAPI documentation
"""

from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import get_settings
from src.interface.api.routes import router
from src.interface.api.middleware import RateLimitMiddleware

logger = logging.getLogger(__name__)

# Track application start time for uptime calculation
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle manager."""
    global _start_time
    _start_time = time.time()
    settings = get_settings()
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} "
        f"[{settings.environment.value}]"
    )
    yield
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Production-grade REST API for the Industrial AI Knowledge Assistant. "
            "Provides RAG-enhanced, domain-specific responses for industrial "
            "automation, building automation, and manufacturing systems."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # ── Middleware ────────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.rate_limit_per_minute,
    )

    # ── Global Exception Handler ─────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):  # noqa: ARG001
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": (
                    str(exc) if settings.debug else "An unexpected error occurred"
                ),
            },
        )

    # ── Routes ───────────────────────────────────────────────────────

    app.include_router(router, prefix="/api/v1")

    return app


def get_uptime() -> float:
    """Get application uptime in seconds."""
    return time.time() - _start_time if _start_time else 0.0
