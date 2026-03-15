This PR makes LangChain imports lazy and adds a small uptime helper to break a circular import between `app.py` and `routes.py`.

These defensive changes allow tests and CI to import modules even when optional runtime dependencies (langchain_community, langchain_core, langchain_openai) are not installed.

Changes:
- src/infrastructure/document_loader.py: lazy imports + safe fallbacks
- src/infrastructure/vector_store.py: lazy imports + tolerant initialization
- src/infrastructure/llm_client.py: lazy imports + dummy LLM fallback
- src/interface/api/uptime.py: new helper module
- src/interface/api/app.py, src/interface/api/routes.py: updated to use uptime helper

Why: Prevents import-time failures in CI/test environments that don't have certain optional packages installed, stabilizing test collection and early pipeline stages.

Notes:
- All formatting and lint/type checks were run locally: `black --check`, `isort --check-only`, `flake8`, `mypy`, and integration tests.
- If you prefer a different base branch, change `--base` when creating the PR.
