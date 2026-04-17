from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fastapi import UploadFile

from ..db.json_jobs import JsonJobStore
from .audio_io import download_url_to_file, ensure_wav
from .model_runtime import MODEL
from .pairwise import pairwise_preference


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _bubble_sort_total_comparisons(n: int) -> int:
    return n * (n - 1) // 2 if n > 1 else 0


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _paths_from_item_dicts(rows: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for it in rows:
        for key in ("original_path", "wav_path"):
            v = it.get(key)
            if isinstance(v, str) and v:
                paths.append(Path(v))
    return paths


def _unlink_audio_files(paths: list[Path]) -> None:
    for p in paths:
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass


async def _delete_job_audio_files(
    store: JsonJobStore,
    job_id: str,
    items: list[dict[str, Any]] | None,
) -> None:
    """Remove downloaded / uploaded / transcoded audio files to save disk space."""
    paths: list[Path] = []
    if items is not None:
        paths = _paths_from_item_dicts(items)
    else:
        doc = await store.get_job(job_id)
        if doc:
            for p in doc.get("temp_paths") or []:
                if isinstance(p, str) and p:
                    paths.append(Path(p))
            paths.extend(_paths_from_item_dicts(list(doc.get("items") or [])))
    paths = _dedupe_paths(paths)
    if paths:
        await asyncio.to_thread(_unlink_audio_files, paths)


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
    prepare_parallel: int,
) -> list[dict[str, Any]]:
    workers = max(1, min(int(prepare_parallel), 32))
    await _update_job(
        store,
        job_id,
        {
            "status": "running",
            "phase": "prepare",
            "message": f"Downloading / saving inputs (prepare_parallel={workers})",
            "prepare_parallel": workers,
            "started_at": _utcnow(),
        },
    )

    items: list[dict[str, Any]] = []
    temp_paths: list[Path] = []
    sem = asyncio.Semaphore(workers)

    async def _prepare_one_url(idx: int, url: str) -> tuple[int, dict[str, Any], list[Path]]:
        async with sem:
            stem = f"url_{idx:04d}"
            src = await asyncio.to_thread(
                download_url_to_file, url, job_dir / "download", stem
            )
            wav = await asyncio.to_thread(ensure_wav, src)
        extras: list[Path] = []
        if wav != src:
            extras.append(wav)
        item_id = str(uuid.uuid4())
        row = {
            "id": item_id,
            "label": stem,
            "source": "url",
            "url": url,
            "original_path": str(src),
            "wav_path": str(wav),
        }
        return (idx, row, extras)

    if urls:
        url_results = await asyncio.gather(
            *(_prepare_one_url(i, u) for i, u in enumerate(urls))
        )
        for idx, row, extras in sorted(url_results, key=lambda t: t[0]):
            items.append(row)
            temp_paths.extend(extras)

    if uploads:
        up_dir = job_dir / "upload"
        up_dir.mkdir(parents=True, exist_ok=True)

        async def _prepare_one_upload(idx: int, up: UploadFile) -> tuple[int, dict[str, Any], list[Path]]:
            async with sem:
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
            extras: list[Path] = []
            if wav != dest:
                extras.append(wav)
            item_id = str(uuid.uuid4())
            row = {
                "id": item_id,
                "label": safe,
                "source": "upload",
                "url": None,
                "original_path": str(dest),
                "wav_path": str(wav),
            }
            return (idx, row, extras)

        up_results = await asyncio.gather(
            *(_prepare_one_upload(i, up) for i, up in enumerate(uploads))
        )
        for idx, row, extras in sorted(up_results, key=lambda t: t[0]):
            items.append(row)
            temp_paths.extend(extras)

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


def _odd_even_sort_sync(
    *,
    items: list[dict[str, Any]],
    compare_pair: Callable[[str, str], int],
    parallel: int,
) -> list[dict[str, Any]]:
    """Odd-even transposition sort; within each phase, disjoint pairs compare in parallel."""
    arr = list(items)
    n = len(arr)
    if n <= 1:
        return arr
    workers = max(1, min(int(parallel), 32))

    def _run_phase_batch(snap: list[tuple[int, str, str]]) -> dict[int, int]:
        if not snap:
            return {}
        use = min(workers, len(snap))
        if use <= 1:
            prefs: dict[int, int] = {}
            for j, left, right in snap:
                prefs[j] = compare_pair(left, right)
            return prefs
        prefs = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=use) as ex:
            fut_to_j = {ex.submit(compare_pair, left, right): j for j, left, right in snap}
            for fut in concurrent.futures.as_completed(fut_to_j):
                j = fut_to_j[fut]
                prefs[j] = fut.result()
        return prefs

    while True:
        changed = False
        for start in (1, 0):
            pairs_idx = [j for j in range(start, n - 1, 2)]
            if not pairs_idx:
                continue
            snap = [(j, arr[j]["wav_path"], arr[j + 1]["wav_path"]) for j in pairs_idx]
            prefs = _run_phase_batch(snap)
            for j in pairs_idx:
                if prefs[j] < 0:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    changed = True
        if not changed:
            break
    return arr


async def run_rank_job(
    *,
    store: JsonJobStore,
    job_id: str,
    settings: Any,
    target_text: str,
    urls: list[str],
    uploads: list[UploadFile] | None,
    pairwise_parallel: int,
    prepare_parallel: int,
) -> None:
    job_dir = settings.job_files_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] | None = None

    try:
        items = await prepare_job_inputs(
            store=store,
            job_id=job_id,
            job_dir=job_dir,
            target_text=target_text,
            urls=urls,
            uploads=uploads,
            prepare_parallel=prepare_parallel,
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
        parallel = max(1, min(int(pairwise_parallel), 32))
        await _update_job(
            store,
            job_id,
            {
                "phase": "sort",
                "message": f"Odd-even sort (pairwise_parallel={parallel})",
                "n_items": n,
                "comparisons_total": total_cmp,
                "comparisons_done": 0,
                "progress": 0.0,
                "pairwise_parallel": parallel,
            },
        )

        model, processor = MODEL.get()
        done = 0
        done_lock = threading.Lock()

        def compare_pair(left_wav: str, right_wav: str) -> int:
            nonlocal done
            if parallel <= 1:
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
            else:
                with MODEL.context():
                    pref = pairwise_preference(
                        processor,
                        model,
                        is_omni=not settings.thinker,
                        max_new_tokens=None,
                        target_text=target_text,
                        left_wav=left_wav,
                        right_wav=right_wav,
                    )
            with done_lock:
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
                with done_lock:
                    d = done
                await _update_job(
                    store,
                    job_id,
                    {
                        "comparisons_done": d,
                        "progress": (d / total_cmp) if total_cmp else 1.0,
                    },
                )

        reporter = asyncio.create_task(progress_reporter())
        try:

            def runner() -> list[dict[str, Any]]:
                return _odd_even_sort_sync(
                    items=items,
                    compare_pair=compare_pair,
                    parallel=parallel,
                )

            ranked = await asyncio.to_thread(runner)
        finally:
            reporter.cancel()
            # CancelledError is BaseException, not Exception — must suppress or background task errors.
            with contextlib.suppress(asyncio.CancelledError):
                await reporter

        ranked_ids = [it["id"] for it in ranked]
        ranked_items_public = [
            {
                "id": it["id"],
                "label": it.get("label"),
                "source": it.get("source"),
                "url": it.get("url"),
            }
            for it in ranked
        ]
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
                "ranked_items": ranked_items_public,
                "items": [],
                "temp_paths": [],
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
        with contextlib.suppress(Exception):
            await _delete_job_audio_files(store, job_id, items)
        shutil.rmtree(job_dir, ignore_errors=True)
