from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class JobCreateRequest(BaseModel):
    target_text: str = Field(..., min_length=1)
    audio_url_a: HttpUrl
    audio_url_b: HttpUrl
    num_of_generation: int = Field(10, ge=1, le=64)


class JobProgress(BaseModel):
    percent: float = 0.0
    step: Literal[
        "queued",
        "downloading",
        "loading_audio",
        "generating",
        "aggregating",
        "done",
        "failed",
    ] = "queued"
    message: str = ""
    current: Optional[int] = None
    total: Optional[int] = None


class JobCreateResponse(BaseModel):
    task_id: str


class JobStatusResponse(BaseModel):
    model_config = {"extra": "ignore"}

    task_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: JobProgress
    request: dict[str, Any]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    cache_paths: Optional[dict[str, str]] = None
    created_at: str
    updated_at: str
