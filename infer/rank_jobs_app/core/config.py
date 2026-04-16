from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    mongo_uri: str
    mongo_db: str
    mongo_collection: str
    job_files_root: Path
    model_path: str
    cuda_device_raw: str | None
    thinker: bool


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[3]
    default_jobs = repo_root / "data" / "rank_jobs"
    return Settings(
        mongo_uri=os.environ.get("SPEECHJUDGE_MONGO_URI", "mongodb://127.0.0.1:27017"),
        mongo_db=os.environ.get("SPEECHJUDGE_MONGO_DB", "speechjudge"),
        mongo_collection=os.environ.get("SPEECHJUDGE_MONGO_COLLECTION", "rank_jobs"),
        job_files_root=Path(
            os.environ.get("SPEECHJUDGE_RANK_JOB_DIR", str(default_jobs))
        ).expanduser(),
        model_path=os.environ.get("SPEECHJUDGE_MODEL_PATH", "pretrained/SpeechJudge-GRM"),
        cuda_device_raw=os.environ.get("SPEECHJUDGE_CUDA_DEVICE"),
        thinker=os.environ.get("SPEECHJUDGE_THINKER", "").lower() in {"1", "true", "yes"},
    )
