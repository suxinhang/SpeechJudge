from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ...services.rank_worker import run_rank_job


class CreateJobResponse(BaseModel):
    job_id: str = Field(..., description="Mongo ObjectId string")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_jobs_router(*, coll, settings) -> APIRouter:
    router = APIRouter()

    @router.post("/jobs/rank", response_model=CreateJobResponse)
    async def create_rank_job(
        background_tasks: BackgroundTasks,
        target_text: str = Form(..., min_length=1),
        urls_json: str | None = Form(
            default=None,
            description='JSON array of strings, e.g. ["https://a/a.mp3","https://b/b.wav"]',
        ),
        audio_files: list[UploadFile] | None = File(default=None),
    ) -> CreateJobResponse:
        urls: list[str] = []
        if urls_json:
            try:
                parsed = json.loads(urls_json)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"urls_json is not valid JSON: {exc}") from exc
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise HTTPException(status_code=400, detail="urls_json must be a JSON array of strings")
            urls = [u.strip() for u in parsed if u.strip()]

        if not urls and not audio_files:
            raise HTTPException(
                status_code=400,
                detail="Provide urls_json and/or audio_files (multipart uploads).",
            )

        doc: dict[str, Any] = {
            "status": "queued",
            "phase": "queued",
            "message": "Queued",
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "target_text": target_text,
            "urls": urls,
            "n_urls": len(urls),
            "n_uploads": len(audio_files or []),
        }
        res = await coll.insert_one(doc)
        job_id = str(res.inserted_id)

        background_tasks.add_task(
            run_rank_job,
            coll=coll,
            job_id=job_id,
            settings=settings,
            target_text=target_text,
            urls=urls,
            uploads=audio_files,
        )
        return CreateJobResponse(job_id=job_id)

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            oid = ObjectId(job_id)
        except InvalidId as exc:
            raise HTTPException(status_code=400, detail="invalid job id") from exc

        doc = await coll.find_one({"_id": oid})
        if doc is None:
            raise HTTPException(status_code=404, detail="job not found")
        doc["job_id"] = str(doc.pop("_id"))
        return doc

    return router
