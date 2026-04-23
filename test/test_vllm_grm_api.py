"""
Smoke test for vLLM GRM FastAPI: reads URL list from testdate.txt, submits one job
(first two URLs as Output A / Output B), polls by task_id until terminal state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

DEFAULT_BASE = "http://127.0.0.1:8000"
DEFAULT_TARGET = "smoke triplet screen"


def load_urls(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    if len(lines) < 2:
        raise SystemExit(f"Need at least 2 URLs in {path}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url",
        default=DEFAULT_BASE,
        help=f"API root (default: {DEFAULT_BASE})",
    )
    ap.add_argument(
        "--data-file",
        type=Path,
        default=Path(__file__).resolve().parent / "testdate.txt",
        help="Plain text: one audio URL per line (default: test/testdate.txt)",
    )
    ap.add_argument(
        "--target-text",
        default=DEFAULT_TARGET,
        help="Target text passed to GRM (default: smoke triplet screen)",
    )
    ap.add_argument(
        "--num-of-generation",
        type=int,
        default=2,
        help="Inference-time scaling count (default 2 for quicker smoke; use 10 for official)",
    )
    ap.add_argument("--poll-interval", type=float, default=2.0)
    args = ap.parse_args()

    urls = load_urls(args.data_file)
    url_a, url_b = urls[0], urls[1]

    payload = {
        "target_text": args.target_text,
        "audio_url_a": url_a,
        "audio_url_b": url_b,
        "num_of_generation": args.num_of_generation,
    }

    base = args.base_url.rstrip("/")
    with httpx.Client(timeout=60.0) as client:
        r = client.get(f"{base}/health")
        r.raise_for_status()
        print("[health]", r.json())

        r = client.post(f"{base}/jobs", json=payload)
        r.raise_for_status()
        task_id = r.json()["task_id"]
        print("[create]", task_id)

        while True:
            r = client.get(f"{base}/jobs/{task_id}")
            r.raise_for_status()
            body = r.json()
            prog = body.get("progress", {})
            print(
                json.dumps(
                    {
                        "status": body.get("status"),
                        "percent": prog.get("percent"),
                        "step": prog.get("step"),
                        "message": prog.get("message"),
                        "current": prog.get("current"),
                        "total": prog.get("total"),
                    },
                    ensure_ascii=False,
                )
            )
            st = body.get("status")
            if st in ("completed", "failed"):
                print("[final]", json.dumps(body, ensure_ascii=False, indent=2))
                if st == "failed":
                    sys.exit(1)
                return
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
