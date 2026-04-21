from __future__ import annotations

import asyncio
import contextlib
import functools
import shutil

import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fastapi import UploadFile

from ..core.ranking import (
    ALGORITHM_FULL_PAIRWISE,
    PHASE_FULL,
    PHASE_CHALLENGE,
    PHASE_EXPLOIT,
    PHASE_EXPLORE,
    PHASE_TOP_K,
    RankingConfig,
    RankingItem,
    estimate_total_budget,
    phase_budgets,
)
from ..db.json_jobs import JsonJobStore
from .audio_io import download_url_to_file, ensure_wav
from .model_runtime import MODEL
from .pairwise import pairwise_preferences_batched
from .ranking_engine import RankingEngine


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    prepare_download_attempts: int,
    prepare_decode_attempts: int,
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
            dl = functools.partial(
                download_url_to_file,
                url,
                job_dir / "download",
                stem,
                max_attempts=prepare_download_attempts,
            )
            src = await asyncio.to_thread(dl)
            wav = await asyncio.to_thread(
                functools.partial(ensure_wav, max_attempts=prepare_decode_attempts), src
            )
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
                wav = await asyncio.to_thread(
                    functools.partial(ensure_wav, max_attempts=prepare_decode_attempts),
                    dest,
                )
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


def _phase_message(phase: str, *, top_k: int) -> str:
    if phase == PHASE_FULL:
        return "Full compare: compare every unique pair once"
    if phase == PHASE_EXPLORE:
        return "Exploration: broad pair coverage"
    if phase == PHASE_EXPLOIT:
        return "Exploitation: compare nearby Elo ratings"
    if phase == PHASE_CHALLENGE:
        return f"Challenge refine: promote strong items below top {top_k}"
    if phase == PHASE_TOP_K:
        return f"Top-K refine: reinforce top {top_k} boundary"
    return "Ranking"


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
    prepare_download_attempts: int,
    prepare_decode_attempts: int,
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
            prepare_download_attempts=prepare_download_attempts,
            prepare_decode_attempts=prepare_decode_attempts,
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

        parallel = max(1, min(int(pairwise_parallel), 32))
        rank_config = RankingConfig(
            algorithm=str(getattr(settings, "rank_algorithm", ALGORITHM_FULL_PAIRWISE)),
            top_k=min(max(1, int(getattr(settings, "rank_top_k", 20))), n),
            budget_multiplier=float(getattr(settings, "rank_budget_multiplier", 2.0)),
            neighbor_window=int(getattr(settings, "rank_neighbor_window", 4)),
            max_pair_repeats=int(getattr(settings, "rank_max_pair_repeats", 3)),
        )
        total_cmp = estimate_total_budget(n, rank_config)
        strategy_message = (
            f"Full pairwise ranking ({total_cmp} unique comparisons)"
            if rank_config.algorithm == ALGORITHM_FULL_PAIRWISE
            else f"Phased Elo ranking (estimated comparisons<={total_cmp})"
        )
        await _update_job(
            store,
            job_id,
            {
                "phase": "sort",
                "message": strategy_message,
                "n_items": n,
                "comparisons_total": total_cmp,
                "comparisons_done": 0,
                "progress": 0.0,
                "pairwise_parallel": parallel,
                "ranking_strategy": {
                    "algorithm": rank_config.algorithm,
                    "top_k": rank_config.top_k,
                    "budget_multiplier": rank_config.budget_multiplier,
                    "neighbor_window": rank_config.neighbor_window,
                    "max_pair_repeats": rank_config.max_pair_repeats,
                },
            },
        )

        model, processor = MODEL.get()
        phase_plan = phase_budgets(n, rank_config)
        phase_order = list(phase_plan)
        progress_state: dict[str, Any] = {
            "phase": phase_order[0] if phase_order else "sort",
            "done": 0,
            "total": total_cmp,
            "phase_counts": {phase: 0 for phase in phase_plan},
        }
        progress_lock = threading.Lock()

        ranking_items = [
            RankingItem(
                id=str(it["id"]),
                wav_path=str(it["wav_path"]),
                label=it.get("label"),
                source=it.get("source"),
                url=it.get("url"),
            )
            for it in items
        ]
        engine = RankingEngine(rank_config)

        def compare_batch(batch: list[tuple[RankingItem, RankingItem]]) -> list[int]:
            wav_pairs = [(left.wav_path, right.wav_path) for left, right in batch]
            with MODEL.infer_lock(), MODEL.context():
                return pairwise_preferences_batched(
                    processor,
                    model,
                    is_omni=not settings.thinker,
                    max_new_tokens=None,
                    target_text=target_text,
                    pairs=wav_pairs,
                )

        def on_progress(snapshot) -> None:
            with progress_lock:
                progress_state["phase"] = snapshot.phase
                progress_state["done"] = snapshot.comparisons_done
                progress_state["total"] = snapshot.comparisons_total
                progress_state["phase_counts"] = dict(snapshot.phase_comparisons)

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
                with progress_lock:
                    current_phase = str(progress_state["phase"])
                    done = int(progress_state["done"])
                    total = int(progress_state["total"])
                    phase_counts = dict(progress_state["phase_counts"])
                await _update_job(
                    store,
                    job_id,
                    {
                        "phase": current_phase,
                        "message": _phase_message(current_phase, top_k=rank_config.top_k),
                        "comparisons_done": done,
                        "comparisons_total": total,
                        "phase_comparisons": phase_counts,
                        "progress": (done / total) if total else 1.0,
                    },
                )

        reporter = asyncio.create_task(progress_reporter())
        try:

            def runner():
                return engine.run(
                    items=ranking_items,
                    compare_batch=compare_batch,
                    batch_size=parallel,
                    progress_callback=on_progress,
                )

            ranked = await asyncio.to_thread(runner)
        finally:
            reporter.cancel()
            # CancelledError is BaseException, not Exception — must suppress or background task errors.
            with contextlib.suppress(asyncio.CancelledError):
                await reporter

        ranked_ids = [it.item.id for it in ranked.items]
        ranked_items_public = [
            {
                "id": it.item.id,
                "label": it.item.label,
                "source": it.item.source,
                "url": it.item.url,
                "rating": round(it.rating, 3),
                "comparisons": it.comparisons,
                "ties": it.ties,
            }
            for it in ranked.items
        ]
        await _update_job(
            store,
            job_id,
            {
                "status": "succeeded",
                "phase": "done",
                "message": "Completed",
                "comparisons_done": ranked.comparisons_done,
                "comparisons_total": ranked.comparisons_total,
                "phase_comparisons": ranked.phase_comparisons,
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
