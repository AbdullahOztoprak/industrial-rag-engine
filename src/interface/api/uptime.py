"""
Uptime helper for the API.

Stores the application start time and exposes a getter for uptime.
This decouples uptime tracking from the FastAPI app module to avoid
circular imports between app and routes.
"""

from __future__ import annotations

import time

_start_time: float = 0.0


def set_start_time(ts: float) -> None:
    global _start_time
    _start_time = float(ts)


def get_uptime() -> float:
    """Return uptime in seconds since start time was set.

    If the start time hasn't been set yet, returns 0.0.
    """
    return time.time() - _start_time if _start_time else 0.0
