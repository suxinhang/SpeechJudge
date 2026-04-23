from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from vllm_grm_api.app.api.routes.jobs import router as jobs_router
from vllm_grm_api.app.core.config import ensure_data_dirs
from vllm_grm_api.app.services.vllm_compare import load_model, resolve_model_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    model_path = resolve_model_path()
    processor, llm, sampling_params = load_model(model_path)
    app.state.processor = processor
    app.state.llm = llm
    app.state.sampling_params = sampling_params
    yield


app = FastAPI(
    title="SpeechJudge GRM vLLM Job API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(jobs_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
