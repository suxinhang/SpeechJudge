"""File-backed job store (one JSON file per job, no database server)."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

_DEFAULT = str


class JsonJobStore:
    """Persist rank job documents under ``root_dir / {job_id}.json``."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    def _path(self, job_id: str) -> Path:
        return self.root_dir / f"{job_id}.json"

    async def _lock_for(self, job_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            if job_id not in self._locks:
                self._locks[job_id] = asyncio.Lock()
            return self._locks[job_id]

    @staticmethod
    def _atomic_write(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        text = json.dumps(data, ensure_ascii=False, indent=2, default=_DEFAULT)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    async def insert_job(self, doc: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        out = {**doc, "job_id": job_id}
        lock = await self._lock_for(job_id)
        async with lock:
            await asyncio.to_thread(self._atomic_write, self._path(job_id), out)
        return job_id

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        lock = await self._lock_for(job_id)
        async with lock:
            return await asyncio.to_thread(self._read, self._path(job_id))

    async def update_job(self, job_id: str, patch: dict[str, Any]) -> None:
        lock = await self._lock_for(job_id)

        def merge() -> None:
            path = self._path(job_id)
            doc = self._read(path)
            if doc is None:
                raise FileNotFoundError(f"job not found: {job_id}")
            doc.update(patch)
            self._atomic_write(path, doc)

        async with lock:
            await asyncio.to_thread(merge)
