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

# 与 scripts/start_vllm_grm_api_local.sh 中 API_PORT 默认一致
DEFAULT_BASE = "http://127.0.0.1:8001"
DEFAULT_TARGET = "smoke triplet screen"

# 其它 SpeechJudge 网关的 /health 常带这些字段；vllm_grm_api 只有 {"status": "ok"}。
_WRONG_HEALTH_MARKERS = (
    "default_mode",
    "model_path",
    "loaded_at",
    "recommended_request_max_new_tokens",
    "cuda_device",
)


def assert_vllm_grm_job_api_health(client: httpx.Client, base: str) -> None:
    try:
        r = client.get(f"{base}/health")
    except httpx.ConnectError as e:
        raise SystemExit(
            f"Cannot connect to {base}/health ({e}).\n"
            "Nothing is listening on that host:port (WinError 10061 = connection refused).\n\n"
            "Start vllm_grm_api first, e.g. in Git Bash from repo root:\n"
            "  bash scripts/start_vllm_grm_api_local.sh\n"
            "(Port is set in that script as API_PORT; match --base-url here.)\n"
            "Then:\n"
            f"  python test/test_vllm_grm_api.py --base-url {base}\n"
        ) from e
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise SystemExit(f"/health returned non-object: {data!r}")
    if data.get("status") == "ok" and not any(k in data for k in _WRONG_HEALTH_MARKERS):
        print("[health]", data)
        return
    print("[health]", data, file=sys.stderr)
    raise SystemExit(
        "This server is not vllm_grm_api (no SpeechJudge GRM Job API on this URL).\n"
        "/jobs will 404. Your port is likely another app (e.g. rank / inference on :8000).\n\n"
        "Fix: start vllm_grm_api (port is set in scripts/start_vllm_grm_api_local.sh or start_vllm_grm_api.sh),\n"
        "then pass the same URL, e.g.\n"
        "  bash scripts/start_vllm_grm_api_local.sh\n"
        "  python test/test_vllm_grm_api.py --base-url http://127.0.0.1:8001\n"
    )


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
        assert_vllm_grm_job_api_health(client, base)

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
