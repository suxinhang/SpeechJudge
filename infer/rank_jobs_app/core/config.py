from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Rank jobs service settings."""

    jobs_state_dir: Path
    job_files_root: Path
    model_path: str
    cuda_device_raw: str | None
    thinker: bool


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[3]
    default_jobs = repo_root / "data" / "rank_jobs"
    job_files_root = Path(
        os.environ.get("SPEECHJUDGE_RANK_JOB_DIR", str(default_jobs))
    ).expanduser()
    default_state = job_files_root / "_json_jobs"
    jobs_state_dir = Path(
        os.environ.get("SPEECHJUDGE_RANK_JOBS_JSON_DIR", str(default_state))
    ).expanduser()
    return Settings(
        jobs_state_dir=jobs_state_dir,
        job_files_root=job_files_root,
        model_path=os.environ.get("SPEECHJUDGE_MODEL_PATH", "pretrained/SpeechJudge-GRM"),
        cuda_device_raw=os.environ.get("SPEECHJUDGE_CUDA_DEVICE"),
        thinker=os.environ.get("SPEECHJUDGE_THINKER", "").lower() in {"1", "true", "yes"},
    )
