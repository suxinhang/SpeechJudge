"""Random triplet screening: shuffle inputs, groups of 3, round-robin pairwise within each group, pick one winner."""

from __future__ import annotations

import asyncio
import contextlib
import random
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ..db.json_jobs import JsonJobStore
from .model_runtime import MODEL
from .pairwise import pairwise_preferences_batched
from .rank_worker import _delete_job_audio_files, _update_job, prepare_job_inputs


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _public_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "label": row.get("label"),
        "source": row.get("source"),
        "url": row.get("url"),
    }


def shuffle_into_groups(
    items: list[dict[str, Any]], *, seed: int, group_size: int = 3
) -> list[list[dict[str, Any]]]:
    shuffled = list(items)
    rng = random.Random(int(seed))
    rng.shuffle(shuffled)
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    return [shuffled[i : i + group_size] for i in range(0, len(shuffled), group_size)]


def _pick_winner_index(wins: list[float], member_ids: list[str]) -> int:
    best = max(wins)
    tied = [i for i, w in enumerate(wins) if w == best]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda i: member_ids[i])


def _run_pairwise_batch(
    *,
    target_text: str,
    pairs: list[tuple[str, str]],
    thinker: bool,
) -> list[int]:
    if not pairs:
        return []
    model, processor = MODEL.get()
    with MODEL.infer_lock(), MODEL.context():
        return pairwise_preferences_batched(
            processor,
            model,
            is_omni=not thinker,
            max_new_tokens=None,
            target_text=target_text,
            pairs=pairs,
        )


async def _run_pairs_one_at_a_time(
    *,
    target_text: str,
    pairs: list[tuple[str, str]],
    thinker: bool,
) -> list[int]:
    """One logical pair per GPU forward.

    Batching multiple long-audio A/B conversations in ``pairwise_preferences_batched`` can
    allocate tens of GiB VRAM; triplet_screen therefore compares sequentially by default.
    """
    outcomes: list[int] = []
    for pair in pairs:
        part = await asyncio.to_thread(
            _run_pairwise_batch,
            target_text=target_text,
            pairs=[pair],
            thinker=thinker,
        )
        outcomes.extend(part)
    return outcomes


async def run_triplet_screen_job(
    *,
    store: JsonJobStore,
    job_id: str,
    settings: Any,
    target_text: str,
    urls: list[str],
    uploads: list[UploadFile] | None,
    prepare_parallel: int,
    prepare_download_attempts: int,
    prepare_decode_attempts: int,
    shuffle_seed: int | None,
    group_size: int,
) -> None:
    job_dir = Path(settings.job_files_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] | None = None
    seed_used = int(shuffle_seed) if shuffle_seed is not None else secrets.randbits(31)

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

        gs = max(1, min(int(group_size), 64))
        groups = shuffle_into_groups(items, seed=seed_used, group_size=gs)
        n_groups = len(groups)

        total_cmp = 0
        for g in groups:
            m = len(g)
            if m <= 1:
                continue
            total_cmp += m * (m - 1) // 2

        await _update_job(
            store,
            job_id,
            {
                "job_kind": "triplet_screen",
                "phase": "triplet_screen",
                "message": (
                    f"Random groups of {gs}, round-robin within each group ({n_groups} groups, "
                    f"{total_cmp} pairwise compares; one model forward per pair to limit VRAM)"
                ),
                "n_items": n,
                "triplet_screen": {
                    "shuffle_seed": seed_used,
                    "group_size": gs,
                    "n_groups": n_groups,
                    "rule": "within each group: all pairs compared once; wins (+0.5 each side on model tie); highest wins wins group; ties broken by lexicographically smallest item id",
                },
                "comparisons_total": total_cmp,
                "comparisons_done": 0,
                "progress": 0.0,
            },
        )

        thinker = bool(getattr(settings, "thinker", False))
        winners: list[dict[str, Any]] = []
        eliminated: list[dict[str, Any]] = []
        groups_out: list[dict[str, Any]] = []
        done_cmp = 0

        for gi, group in enumerate(groups):
            m = len(group)
            member_ids = [str(x["id"]) for x in group]
            members_pub = [_public_item(x) for x in group]
            comps_log: list[dict[str, Any]] = []

            if m == 1:
                widx = 0
                wins = [0.0]
            elif m == 2:
                wav_pairs = [(str(group[0]["wav_path"]), str(group[1]["wav_path"]))]
                outcomes = await _run_pairs_one_at_a_time(
                    target_text=target_text,
                    pairs=wav_pairs,
                    thinker=thinker,
                )
                done_cmp += len(wav_pairs)
                o = outcomes[0]
                comps_log.append(
                    {
                        "left_id": member_ids[0],
                        "right_id": member_ids[1],
                        "preference": o,
                        "verdict": "left" if o > 0 else ("right" if o < 0 else "tie"),
                    }
                )
                wins = [0.0, 0.0]
                if o > 0:
                    wins[0] = 1.0
                elif o < 0:
                    wins[1] = 1.0
                else:
                    wins[0] = wins[1] = 0.5
                widx = _pick_winner_index(wins, member_ids)
            else:
                pair_idx = [(i, j) for i in range(m) for j in range(i + 1, m)]
                wav_pairs = [
                    (str(group[i]["wav_path"]), str(group[j]["wav_path"])) for i, j in pair_idx
                ]
                outcomes = await _run_pairs_one_at_a_time(
                    target_text=target_text,
                    pairs=wav_pairs,
                    thinker=thinker,
                )
                done_cmp += len(wav_pairs)
                wins = [0.0] * m
                for (i, j), o in zip(pair_idx, outcomes):
                    comps_log.append(
                        {
                            "left_id": member_ids[i],
                            "right_id": member_ids[j],
                            "preference": o,
                            "verdict": "left" if o > 0 else ("right" if o < 0 else "tie"),
                        }
                    )
                    if o > 0:
                        wins[i] += 1.0
                    elif o < 0:
                        wins[j] += 1.0
                    else:
                        wins[i] += 0.5
                        wins[j] += 0.5
                widx = _pick_winner_index(wins, member_ids)

            for k, row in enumerate(group):
                pub = _public_item(row)
                entry = {
                    "group_index": gi,
                    "wins_in_group": round(float(wins[k]), 3),
                    **pub,
                }
                if k == widx:
                    winners.append(entry)
                else:
                    eliminated.append({**entry, "lost_to_winner_id": member_ids[widx]})

            groups_out.append(
                {
                    "group_index": gi,
                    "member_ids": member_ids,
                    "members": members_pub,
                    "winner_id": member_ids[widx],
                    "wins_in_group": [round(float(w), 3) for w in wins],
                    "comparisons": comps_log,
                }
            )

            await _update_job(
                store,
                job_id,
                {
                    "comparisons_done": done_cmp,
                    "progress": (done_cmp / total_cmp) if total_cmp else 1.0,
                },
            )

        done_payload: dict[str, Any] = {
            "status": "succeeded",
            "phase": "done",
            "message": "Triplet screen completed",
            "job_kind": "triplet_screen",
            "comparisons_done": done_cmp,
            "comparisons_total": total_cmp,
            "progress": 1.0,
            "triplet_screen": {
                "shuffle_seed": seed_used,
                "group_size": gs,
                "n_groups": n_groups,
                "n_winners": len(winners),
                "n_eliminated": len(eliminated),
            },
            "winners": winners,
            "eliminated": eliminated,
            "groups": groups_out,
            "items": [],
            "temp_paths": [],
            "finished_at": _utcnow(),
        }
        await _update_job(store, job_id, done_payload)
    except Exception as exc:
        await _update_job(
            store,
            job_id,
            {
                "status": "failed",
                "phase": "failed",
                "message": "Triplet screen job failed",
                "error": str(exc),
                "finished_at": _utcnow(),
            },
        )
    finally:
        with contextlib.suppress(Exception):
            await _delete_job_audio_files(store, job_id, items)
        shutil.rmtree(job_dir, ignore_errors=True)
        with contextlib.suppress(Exception):
            MODEL.release_ephemeral_cuda_cache()
