from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.config import get_tasks_dir


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_task_id() -> str:
    return str(uuid.uuid4())


def _task_path(task_id: str) -> Path:
    safe = task_id.replace("/", "").replace("\\", "").replace("..", "")
    return get_tasks_dir() / f"{safe}.json"


def write_task_atomic(task_id: str, payload: dict[str, Any]) -> None:
    path = _task_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = _utc_now_iso()
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_task(task_id: str) -> Optional[dict[str, Any]]:
    path = _task_path(task_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def create_queued_job(
    request_body: dict[str, Any],
    task_id_factory: Callable[[], str] = new_task_id,
) -> str:
    task_id = task_id_factory()
    now = _utc_now_iso()
    payload = {
        "task_id": task_id,
        "status": "queued",
        "progress": {
            "percent": 0.0,
            "step": "queued",
            "message": "Waiting for worker",
        },
        "request": request_body,
        "result": None,
        "error": None,
        "cache_paths": None,
        "created_at": now,
        "updated_at": now,
    }
    write_task_atomic(task_id, payload)
    return task_id


def patch_task(task_id: str, mutator: Callable[[dict[str, Any]], None]) -> None:
    data = read_task(task_id)
    if data is None:
        raise KeyError(task_id)
    mutator(data)
    write_task_atomic(task_id, data)
