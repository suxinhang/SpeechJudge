"""Compatibility entrypoint.

Preferred ASGI path:

    cd infer
    python -m uvicorn rank_jobs_app.app.main:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

from .app.main import app  # noqa: F401
