from __future__ import annotations

import asyncio
import contextlib
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ..db.json_jobs import JsonJobStore
from .audio_io import download_url_to_file, ensure_wav
from .model_runtime import MODEL
from .pairwise import pairwise_preference


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _bubble_sort_total_comparisons(n: int) -> int:
    return n * (n - 1) // 2 if n > 1 else 0


async def _update_job(store: JsonJobStore, job_id: str, patch: dict[str, Any]) -> None:
    patch = {**patch, "updated_at": _utcnow()}
    await store.update_job(job_id, patch)


def _safe_filename(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._") or "audio"
    return out[:180]


async def prepare_job_inputs(
    *,
    store: JsonJobStore,
    job_id: str,
    job_dir: Path,
    target_text: str,
    urls: list[str],
    uploads: list[UploadFile] | None,
) -> list[dict[str, Any]]:
    await _update_job(
        store,
        job_id,
        {
            "status": "running",
            "phase": "prepare",
            "message": "Downloading / saving inputs",
            "started_at": _utcnow(),
        },
    )

    items: list[dict[str, Any]] = []
    temp_paths: list[Path] = []

    for idx, url in enumerate(urls):
        stem = f"url_{idx:04d}"
        src = await asyncio.to_thread(download_url_to_file, url, job_dir / "download", stem)
        wav = await asyncio.to_thread(ensure_wav, src)
        if wav != src:
            temp_paths.append(wav)
        item_id = str(uuid.uuid4())
        items.append(
            {
                "id": item_id,
                "label": stem,
                "source": "url",
                "url": url,
                "original_path": str(src),
                "wav_path": str(wav),
            }
        )

    if uploads:
        up_dir = job_dir / "upload"
        up_dir.mkdir(parents=True, exist_ok=True)
        for idx, up in enumerate(uploads):
            raw_name = up.filename or f"upload_{idx:04d}"
            safe = _safe_filename(Path(raw_name).name)
            dest = up_dir / f"{idx:04d}_{safe}"
            with dest.open("wb") as f:
                while True:
                    chunk = await up.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            await up.close()

            wav = await asyncio.to_thread(ensure_wav, dest)
            if wav != dest:
                temp_paths.append(wav)
            item_id = str(uuid.uuid4())
            items.append(
                {
                    "id": item_id,
                    "label": safe,
                    "source": "upload",
                    "url": None,
                    "original_path": str(dest),
                    "wav_path": str(wav),
                }
            )

    await _update_job(
        store,
        job_id,
        {
            "target_text": target_text,
            "items": items,
            "temp_paths": [str(p) for p in temp_paths],
            "phase": "prepare",
            "message": f"Prepared {len(items)} wav inputs",
        },
    )
    return items


def _bubble_sort_sync(
    *,
    items: list[dict[str, Any]],
    on_compare,
) -> list[dict[str, Any]]:
    arr = list(items)
    n = len(arr)
    if n <= 1:
        return arr

    for i in range(n):
        for j in range(0, n - i - 1):
            left = arr[j]["wav_path"]
            right = arr[j + 1]["wav_path"]
            pref = on_compare(left, right)
            if pref < 0:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


async def run_rank_job(
    *,
    store: JsonJobStore,
    job_id: str,
    settings: Any,
    target_text: str,
    urls: list[str],
    uploads: list[UploadFile] | None,
) -> None:
    job_dir = settings.job_files_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        items = await prepare_job_inputs(
            store=store,
            job_id=job_id,
            job_dir=job_dir,
            target_text=target_text,
            urls=urls,
            uploads=uploads,
        )

        n = len(items)
        if n == 0:
            await _update_job(
                store,
                job_id,
                {
                    "status": "failed",
                    "phase": "failed",
                    "message": "No inputs: provide urls_json or audio_files",
                    "finished_at": _utcnow(),
                },
            )
            return

        total_cmp = _bubble_sort_total_comparisons(n)
        await _update_job(
            store,
            job_id,
            {
                "phase": "sort",
                "message": "Bubble sorting (pairwise comparisons)",
                "n_items": n,
                "comparisons_total": total_cmp,
                "comparisons_done": 0,
                "progress": 0.0,
            },
        )

        model, processor = MODEL.get()
        done = 0

        def on_compare(left_wav: str, right_wav: str) -> int:
            nonlocal done
            with MODEL.infer_lock(), MODEL.context():
                pref = pairwise_preference(
                    processor,
                    model,
                    is_omni=not settings.thinker,
                    max_new_tokens=None,
                    target_text=target_text,
                    left_wav=left_wav,
                    right_wav=right_wav,
                )
            done += 1
            return pref

        last_report = {"t": 0.0}

        async def progress_reporter() -> None:
            while True:
                await asyncio.sleep(0.25)
                doc = await store.get_job(job_id)
                if not doc or doc.get("status") not in {"running"}:
                    return
                now = time.time()
                if now - last_report["t"] < 1.0:
                    continue
                last_report["t"] = now
                await _update_job(
                    store,
                    job_id,
                    {
                        "comparisons_done": done,
                        "progress": (done / total_cmp) if total_cmp else 1.0,
                    },
                )

        reporter = asyncio.create_task(progress_reporter())
        try:

            def runner() -> list[dict[str, Any]]:
                return _bubble_sort_sync(items=items, on_compare=on_compare)

            ranked = await asyncio.to_thread(runner)
        finally:
            reporter.cancel()
            with contextlib.suppress(Exception):
                await reporter

        ranked_ids = [it["id"] for it in ranked]
        await _update_job(
            store,
            job_id,
            {
                "status": "succeeded",
                "phase": "done",
                "message": "Completed",
                "comparisons_done": done,
                "comparisons_total": total_cmp,
                "progress": 1.0,
                "ranked_ids": ranked_ids,
                "ranked_items": ranked,
                "finished_at": _utcnow(),
            },
        )
    except Exception as exc:
        await _update_job(
            store,
            job_id,
            {
                "status": "failed",
                "phase": "failed",
                "message": "Job failed",
                "error": str(exc),
                "finished_at": _utcnow(),
            },
        )
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
