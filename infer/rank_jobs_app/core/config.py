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
    #: Max pairwise compares per batched GPU forward in odd-even phases (1 = serial + infer lock).
    pairwise_parallel: int
    #: Max concurrent URL downloads + wav transcodes during job prepare (1 = fully serial).
    prepare_parallel: int
    #: Per-URL HTTP download attempts (each attempt retries transient errors inside audio_io).
    prepare_download_attempts: int
    #: Per-file decode_to_wav attempts (ffmpeg / librosa flakes).
    prepare_decode_attempts: int
    #: Focus set size for refinement is derived from this Top-K target.
    rank_top_k: int
    #: Multiplier applied to the old merge-sort-style baseline budget.
    rank_budget_multiplier: float
    #: Nearby items compared during exploitation.
    rank_neighbor_window: int
    #: Max repeated comparisons kept for an uncertain pair.
    rank_max_pair_repeats: int


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
    raw_pp = os.environ.get("SPEECHJUDGE_PAIRWISE_PARALLEL", "3").strip()
    try:
        pairwise_parallel = int(raw_pp)
    except ValueError:
        pairwise_parallel = 3
    pairwise_parallel = max(1, min(pairwise_parallel, 32))

    raw_prep = os.environ.get("SPEECHJUDGE_PREPARE_PARALLEL", "8").strip()
    try:
        prepare_parallel = int(raw_prep)
    except ValueError:
        prepare_parallel = 8
    prepare_parallel = max(1, min(prepare_parallel, 32))

    raw_pda = os.environ.get("SPEECHJUDGE_PREPARE_DOWNLOAD_ATTEMPTS", "5").strip()
    try:
        prepare_download_attempts = int(raw_pda)
    except ValueError:
        prepare_download_attempts = 5
    prepare_download_attempts = max(1, min(prepare_download_attempts, 15))

    raw_pdec = os.environ.get("SPEECHJUDGE_PREPARE_DECODE_ATTEMPTS", "3").strip()
    try:
        prepare_decode_attempts = int(raw_pdec)
    except ValueError:
        prepare_decode_attempts = 3
    prepare_decode_attempts = max(1, min(prepare_decode_attempts, 10))

    raw_top_k = os.environ.get("SPEECHJUDGE_RANK_TOP_K", "20").strip()
    try:
        rank_top_k = int(raw_top_k)
    except ValueError:
        rank_top_k = 20
    rank_top_k = max(1, min(rank_top_k, 200))

    raw_budget_mult = os.environ.get("SPEECHJUDGE_RANK_BUDGET_MULTIPLIER", "2.0").strip()
    try:
        rank_budget_multiplier = float(raw_budget_mult)
    except ValueError:
        rank_budget_multiplier = 2.0
    rank_budget_multiplier = max(1.0, min(rank_budget_multiplier, 8.0))

    raw_window = os.environ.get("SPEECHJUDGE_RANK_NEIGHBOR_WINDOW", "4").strip()
    try:
        rank_neighbor_window = int(raw_window)
    except ValueError:
        rank_neighbor_window = 4
    rank_neighbor_window = max(1, min(rank_neighbor_window, 20))

    raw_repeats = os.environ.get("SPEECHJUDGE_RANK_MAX_PAIR_REPEATS", "3").strip()
    try:
        rank_max_pair_repeats = int(raw_repeats)
    except ValueError:
        rank_max_pair_repeats = 3
    rank_max_pair_repeats = max(1, min(rank_max_pair_repeats, 10))

    return Settings(
        jobs_state_dir=jobs_state_dir,
        job_files_root=job_files_root,
        model_path=os.environ.get("SPEECHJUDGE_MODEL_PATH", "pretrained/SpeechJudge-GRM"),
        cuda_device_raw=os.environ.get("SPEECHJUDGE_CUDA_DEVICE"),
        thinker=os.environ.get("SPEECHJUDGE_THINKER", "").lower() in {"1", "true", "yes"},
        pairwise_parallel=pairwise_parallel,
        prepare_parallel=prepare_parallel,
        prepare_download_attempts=prepare_download_attempts,
        prepare_decode_attempts=prepare_decode_attempts,
        rank_top_k=rank_top_k,
        rank_budget_multiplier=rank_budget_multiplier,
        rank_neighbor_window=rank_neighbor_window,
        rank_max_pair_repeats=rank_max_pair_repeats,
    )
