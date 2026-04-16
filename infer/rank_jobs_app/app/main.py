from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

_INFER_DIR = Path(__file__).resolve().parents[2]
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))

from ..api.routes.jobs import build_jobs_router
from ..core.config import load_settings
from ..db.json_jobs import JsonJobStore
from ..services.model_runtime import MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    settings.job_files_root.mkdir(parents=True, exist_ok=True)
    settings.jobs_state_dir.mkdir(parents=True, exist_ok=True)

    cuda_raw = settings.cuda_device_raw
    cuda_device = int(cuda_raw) if cuda_raw is not None else None
    MODEL.load_once(model_path=settings.model_path, cuda_device=cuda_device, thinker=settings.thinker)

    store = JsonJobStore(settings.jobs_state_dir)

    app.state.settings = settings
    app.state.jobs_store = store

    app.include_router(build_jobs_router(store=store, settings=settings))

    yield


app = FastAPI(title="SpeechJudge Rank Jobs API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    settings = app.state.settings
    model, _processor = MODEL.get()
    idx = getattr(model, "device", None)
    cuda_index = getattr(idx, "index", None) if idx is not None else None
    return {
        "status": "ready",
        "jobs_backend": "json",
        "jobs_state_dir": str(settings.jobs_state_dir),
        "job_files_root": str(settings.job_files_root),
        "model_path": settings.model_path,
        "cuda_device": cuda_index,
    }
