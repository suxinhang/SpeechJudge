from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from ...schemas import JobCreateRequest, JobCreateResponse, JobStatusResponse
from ...services.job_repository import create_queued_job, read_task
from ...services.job_runner import run_compare_job

router = APIRouter(tags=["jobs"])


@router.post("/jobs", response_model=JobCreateResponse)
def create_job(
    req: JobCreateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JobCreateResponse:
    body = req.model_dump(mode="json")
    task_id = create_queued_job(body)
    background_tasks.add_task(
        run_compare_job,
        task_id,
        request.app.state.processor,
        request.app.state.llm,
        request.app.state.sampling_params,
    )
    return JobCreateResponse(task_id=task_id)


@router.get("/jobs/{task_id}", response_model=JobStatusResponse)
def get_job(task_id: str) -> JobStatusResponse:
    row = read_task(task_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    return JobStatusResponse.model_validate(row)
