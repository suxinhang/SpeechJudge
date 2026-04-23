from __future__ import annotations

import threading
import traceback
from typing import Any

_JOB_LOCK = threading.Lock()

from ..core.config import get_audio_cache_dir
from .audio_fetch import download_audio_to_cache
from .job_repository import patch_task, read_task, write_task_atomic
from .vllm_compare import aggregate_like_official_main, compare_wavs


def _set_progress(
    data: dict[str, Any],
    *,
    percent: float,
    step: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    p: dict[str, Any] = {
        "percent": round(percent, 2),
        "step": step,
        "message": message,
    }
    if extra:
        p.update(extra)
    data["progress"] = p


def run_compare_job(task_id: str, processor: Any, llm: Any, sampling_params: Any) -> None:
    with _JOB_LOCK:
        _run_compare_job_locked(task_id, processor, llm, sampling_params)


def _run_compare_job_locked(
    task_id: str, processor: Any, llm: Any, sampling_params: Any
) -> None:
    row = read_task(task_id)
    if row is None:
        return

    req = row["request"]
    target_text = req["target_text"]
    url_a = str(req["audio_url_a"])
    url_b = str(req["audio_url_b"])
    n_gen = int(req.get("num_of_generation", 10))

    def save(row_: dict[str, Any]) -> None:
        row_["status"] = "running"
        write_task_atomic(task_id, row_)

    try:
        row["status"] = "running"
        _set_progress(
            row,
            percent=5.0,
            step="downloading",
            message="Fetching audio_url_a",
        )
        save(row)

        cache = get_audio_cache_dir()
        path_a = download_audio_to_cache(url_a, cache)

        _set_progress(
            row,
            percent=15.0,
            step="downloading",
            message="Fetching audio_url_b",
        )
        save(row)

        path_b = download_audio_to_cache(url_b, cache)
        row["cache_paths"] = {"wav_a": str(path_a), "wav_b": str(path_b)}
        _set_progress(
            row,
            percent=25.0,
            step="loading_audio",
            message="Preparing vLLM multimodal inputs",
        )
        save(row)

        def on_progress(step: str, extra: dict[str, Any]) -> None:
            def m(d: dict[str, Any]) -> None:
                cur = int(extra.get("current", 0))
                tot = int(extra.get("total", n_gen)) or n_gen
                if step == "loading_audio":
                    pct = 30.0
                elif step == "generating":
                    base = 30.0
                    span = 60.0
                    pct = base + span * (cur / max(tot, 1))
                else:
                    pct = d["progress"].get("percent", 0.0)
                _set_progress(
                    d,
                    percent=min(pct, 95.0),
                    step=step,
                    message=str(extra.get("message", "")),
                    extra={"current": cur, "total": tot} if step == "generating" else None,
                )

            patch_task(task_id, m)

        results = compare_wavs(
            processor,
            llm,
            sampling_params,
            target_text,
            str(path_a),
            str(path_b),
            num_of_generation=n_gen,
            on_progress=on_progress,
        )

        def agg_patch(d: dict[str, Any]) -> None:
            _set_progress(
                d,
                percent=96.0,
                step="aggregating",
                message="Averaging scores like official main_grm_vllm.py",
            )

        patch_task(task_id, agg_patch)

        summary = aggregate_like_official_main(results)

        def done_patch(d: dict[str, Any]) -> None:
            d["status"] = "completed"
            d["result"] = summary
            d["error"] = None
            _set_progress(
                d,
                percent=100.0,
                step="done",
                message="Completed",
            )

        patch_task(task_id, done_patch)

    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        def fail_patch(d: dict[str, Any]) -> None:
            d["status"] = "failed"
            d["error"] = err
            d["result"] = None
            _set_progress(
                d,
                percent=float(d.get("progress", {}).get("percent", 0.0)),
                step="failed",
                message="Job failed (see error)",
            )

        patch_task(task_id, fail_patch)
