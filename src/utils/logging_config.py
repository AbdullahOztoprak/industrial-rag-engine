"""
Structured logging configuration.

Supports JSON format for production (ELK/Grafana compatible)
and human-readable format for development.
"""

from __future__ import annotations

import logging
import json
import sys
from datetime import datetime
from typing import Optional

from src.config.settings import LogLevel


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "created", "relativeCreated",
                    "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                    "pathname", "filename", "module", "levelname", "levelno",
                    "msecs", "thread", "threadName", "process", "processName",
                    "message", "taskName",
                ):
                    log_data[key] = value

        if record.exc_info and record.exc_info[1]:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    log_format: str = "json",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level.
        log_format: "json" for structured logs, "text" for human-readable.
        log_file: Optional file path for log output.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level.value)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
