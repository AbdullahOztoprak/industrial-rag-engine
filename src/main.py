from __future__ import annotations
"""
Industrial AI Knowledge Assistant — Application Entry Point.

Usage:
    # Run Streamlit UI
    streamlit run src/main.py

    # Run FastAPI server
    uvicorn src.interface.api.app:create_app --factory --reload
"""
import sys
from pathlib import Path
# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.settings import get_settings
from src.utils.logging_config import configure_logging


def run_ui() -> None:
    """Launch the Streamlit UI."""
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        log_format="text",  # Human-readable for UI mode
    )
    from src.interface.ui.streamlit_app import main

    main()


def run_api() -> None:
    """Launch the FastAPI server."""
    import uvicorn

    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        log_format=settings.log_format,
        log_file=settings.log_file,
    )

    uvicorn.run(
        "src.interface.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
    )


if __name__ == "__main__":
    # Default to UI mode when run directly
    run_ui()
