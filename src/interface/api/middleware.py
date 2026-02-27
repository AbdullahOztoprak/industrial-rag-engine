"""
Custom middleware for the API layer.
"""

from __future__ import annotations

import time
import logging
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.

    For production, use Redis-backed rate limiting (e.g., slowapi).
    This implementation is suitable for PoC / single-instance deployment.
    """

    def __init__(self, app, max_requests: int = 60) -> None:  # noqa: ANN001
        super().__init__(app)
        self.max_requests = max_requests
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = 60.0  # 1 minute window

        # Clean old entries
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] if now - ts < window
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again later."},
            )

        self._requests[client_ip].append(now)

        # Log request
        start = time.perf_counter()
        response = await call_next(request)
        latency = (time.perf_counter() - start) * 1000

        logger.info(
            "API request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(latency, 2),
                "client": client_ip,
            },
        )

        return response
