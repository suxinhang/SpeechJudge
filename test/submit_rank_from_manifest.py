"""Step 2 only: POST /jobs/rank from manifest written by download_inputs_from_testd1_xlsx.py.

Does **not** read the xlsx or download URLs. Ensures inputs were prepared and checked in step 1.

Example::

    python test/submit_rank_from_manifest.py \\
      --base-url https://since-supporting-edwards-comments.trycloudflare.com
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

from testd1_inputs_lib import load_manifest


def _configure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _submit_multipart(
    base_url: str,
    target_text: str,
    paths: list[tuple[str, Path]],
    submit_timeout: int,
) -> str:
    multipart: list[tuple[str, tuple[str, Any, str]]] = []
    opened: list[Any] = []
    try:
        for upload_name, p in paths:
            f = p.open("rb")
            opened.append(f)
            multipart.append(("audio_files", (upload_name, f, "application/octet-stream")))
        resp = requests.post(
            f"{base_url.rstrip('/')}/jobs/rank",
            data={"target_text": target_text},
            files=multipart,
            timeout=submit_timeout,
        )
    finally:
        for f in opened:
            f.close()
    if resp.status_code >= 400:
        raise RuntimeError(f"POST /jobs/rank failed: HTTP {resp.status_code}, body={resp.text}")
    payload = resp.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"missing job_id: {json.dumps(payload, ensure_ascii=False)}")
    return str(job_id)


def _poll_job(base_url: str, job_id: str, timeout_seconds: int, interval: float) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    url = f"{base_url.rstrip('/')}/jobs/{job_id}"
    last: dict[str, Any] = {}
    while time.time() < deadline:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        last = resp.json()
        status = str(last.get("status", ""))
        phase = str(last.get("phase", ""))
        progress = float(last.get("progress", 0.0) or 0.0)
        print(f"[poll] status={status} phase={phase} progress={progress:.4f}")
        if status in {"succeeded", "failed"}:
            return last
        time.sleep(interval)
    raise TimeoutError(f"job did not finish in {timeout_seconds}s; last keys={list(last)}")


def main() -> int:
    _configure_stdout_utf8()
    p = argparse.ArgumentParser(description="Submit rank job from testd1_rank_inputs.json manifest.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "testd1_rank_inputs.json",
    )
    p.add_argument("--base-url", required=True, help="Rank API base URL")
    p.add_argument("--submit-timeout", type=int, default=1800)
    p.add_argument("--job-timeout", type=int, default=28800)
    p.add_argument("--poll-interval", type=float, default=5.0)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "testd1_rank_result.json",
    )
    args = p.parse_args()

    raw = load_manifest(args.manifest)
    target_text = str(raw["target_text"])
    items_raw = raw["items"]
    paths: list[tuple[str, Path]] = []
    for it in items_raw:
        if not isinstance(it, dict):
            raise ValueError("each manifest.items entry must be an object")
        up = str(it.get("upload_name", ""))
        ps = str(it.get("path", ""))
        if not up or not ps:
            raise ValueError("each item needs upload_name and path")
        path = Path(ps)
        if not path.is_file():
            print(f"[error] manifest path missing: {path}", file=sys.stderr)
            return 1
        if path.stat().st_size <= 0:
            print(f"[error] empty file: {path}", file=sys.stderr)
            return 1
        paths.append((up, path))

    n = len(paths)
    est = raw.get("estimated_pairwise_comparisons", n * (n - 1) // 2 if n > 1 else 0)
    print(f"[info] manifest items={n}, estimated comparisons={est}")

    job_id = _submit_multipart(args.base_url, target_text, paths, args.submit_timeout)
    print(f"[info] job_id={job_id}")
    result = _poll_job(args.base_url, job_id, args.job_timeout, args.poll_interval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] wrote {args.output}")
    if str(result.get("status")) != "succeeded":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
