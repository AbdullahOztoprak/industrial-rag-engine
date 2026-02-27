# ─── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: Production ─────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="Abdullah" \
      description="Industrial AI Knowledge Assistant" \
      version="1.0.0"

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY requirements.txt .
COPY pyproject.toml .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Environment defaults
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Default: run API server
CMD ["python", "-m", "uvicorn", "src.interface.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
