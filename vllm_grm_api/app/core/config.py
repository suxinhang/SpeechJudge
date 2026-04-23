from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def get_repo_root() -> Path:
    return _REPO_ROOT


def get_data_root() -> Path:
    raw = os.environ.get("SPEECHJUDGE_VLLM_API_DATA", "").strip()
    if raw:
        return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    return (_REPO_ROOT / "data" / "vllm_grm_jobs").resolve()


def get_tasks_dir() -> Path:
    return get_data_root() / "tasks"


def get_audio_cache_dir() -> Path:
    return get_data_root() / "audio_cache"


def ensure_data_dirs() -> None:
    get_tasks_dir().mkdir(parents=True, exist_ok=True)
    get_audio_cache_dir().mkdir(parents=True, exist_ok=True)
