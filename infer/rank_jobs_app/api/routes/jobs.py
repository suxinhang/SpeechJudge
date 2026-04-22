from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ...db.json_jobs import JsonJobStore
from ...services.rank_worker import (
    _job_full_pairwise_aggregation,
    _job_rank_algorithm,
    run_rank_job,
)


class CreateJobResponse(BaseModel):
    job_id: str = Field(..., description="UUID string for the rank job")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_jobs_router(*, store: JsonJobStore, settings) -> APIRouter:
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
        pairwise_parallel: int | None = Form(
            default=None,
            description=(
                "Ranking phase: max pairwise compares per batched GPU forward (1–32; 1=serial + infer lock). "
                "Omit to use server default from SPEECHJUDGE_PAIRWISE_PARALLEL."
            ),
        ),
        pairwise_votes_per_pair: int | None = Form(
            default=None,
            description=(
                "Votes per logical pair: 1 = one forward pass per pair; 3 = adaptive 2-of-3. "
                "Omit to use server default from SPEECHJUDGE_PAIRWISE_VOTES_PER_PAIR (default 1)."
            ),
        ),
        prepare_parallel: int | None = Form(
            default=None,
            description=(
                "Prepare phase: max concurrent URL downloads / upload transcodes (1–32). "
                "Omit to use server default from SPEECHJUDGE_PREPARE_PARALLEL."
            ),
        ),
        rank_algorithm: str | None = Form(
            default=None,
            description=(
                "Override ranking mode for this job: `full_pairwise` or `phased_elo` "
                "(aliases: phased, fast). Omit to use server SPEECHJUDGE_RANK_ALGORITHM."
            ),
        ),
        full_pairwise_aggregation: str | None = Form(
            default=None,
            description=(
                "When rank mode is full_pairwise: `round_robin_points` (default), "
                "`bradley_terry` (bt), or `rank_centrality_bt` (rc_bt: Rank Centrality + BT refine, "
                "response lists top_k only). Omit to use SPEECHJUDGE_RANK_FULL_PAIRWISE_AGGREGATION."
            ),
        ),
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

        pp = int(getattr(settings, "pairwise_parallel", 3))
        if pairwise_parallel is not None:
            pp = int(pairwise_parallel)
            if pp < 1 or pp > 32:
                raise HTTPException(
                    status_code=400,
                    detail="pairwise_parallel must be between 1 and 32",
                )
        pp = max(1, min(pp, 32))

        pair_votes = int(getattr(settings, "pairwise_votes_per_pair", 1))
        if pairwise_votes_per_pair is not None:
            pair_votes = int(pairwise_votes_per_pair)
            if pair_votes not in {1, 3}:
                raise HTTPException(
                    status_code=400,
                    detail="pairwise_votes_per_pair must be either 1 or 3",
                )
        pair_votes = 1 if pair_votes == 1 else 3

        prep = int(getattr(settings, "prepare_parallel", 8))
        if prepare_parallel is not None:
            prep = int(prepare_parallel)
            if prep < 1 or prep > 32:
                raise HTTPException(
                    status_code=400,
                    detail="prepare_parallel must be between 1 and 32",
                )
        prep = max(1, min(prep, 32))

        prep_dl = int(getattr(settings, "prepare_download_attempts", 5))
        prep_dec = int(getattr(settings, "prepare_decode_attempts", 3))

        job_rank_algorithm = _job_rank_algorithm(rank_algorithm, settings=settings)
        job_full_pairwise_aggregation = _job_full_pairwise_aggregation(
            full_pairwise_aggregation, settings=settings
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
            "pairwise_parallel": pp,
            "pairwise_votes_per_pair": pair_votes,
            "prepare_parallel": prep,
            "prepare_download_attempts": prep_dl,
            "prepare_decode_attempts": prep_dec,
            "rank_algorithm": job_rank_algorithm,
            "full_pairwise_aggregation": job_full_pairwise_aggregation,
        }
        try:
            job_id = await store.insert_job(doc)
        except OSError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot write job state: {exc}",
            ) from exc

        background_tasks.add_task(
            run_rank_job,
            store=store,
            job_id=job_id,
            settings=settings,
            target_text=target_text,
            urls=urls,
            uploads=audio_files,
            pairwise_parallel=pp,
            pairwise_votes_per_pair=pair_votes,
            prepare_parallel=prep,
            prepare_download_attempts=prep_dl,
            prepare_decode_attempts=prep_dec,
            rank_algorithm=rank_algorithm,
            full_pairwise_aggregation=full_pairwise_aggregation,
        )
        return CreateJobResponse(job_id=job_id)

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            uuid.UUID(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid job id") from exc

        doc = await store.get_job(job_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="job not found")
        return doc

    return router
