"""Smoke test for POST /jobs/triplet_screen (random groups, round-robin within group, winners + eliminated).

  cd test
  python smoke_triplet_screen.py --urls-file date.txt
  python smoke_triplet_screen.py --urls-file date.txt --base-url https://...

  Default base URL: ``http://127.0.0.1:8010/`` (local uvicorn), or override with env
  ``SPEECHJUDGE_SMOKE_BASE_URL`` / CLI ``--base-url``. Ephemeral ``*.trycloudflare.com`` tunnels
  stop resolving once the tunnel process exits; use a live URL or localhost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

# 联调默认：本机 rank API；可用环境变量或 --base-url 覆盖（勿依赖已过期的 trycloudflare 域名）
SMOKE_DEFAULT_BASE_URL = os.environ.get("SPEECHJUDGE_SMOKE_BASE_URL", "https://fossil-refine-prize-allowing.trycloudflare.com/")


def load_urls_from_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    seen: set[str] = set()
    out: list[str] = []
    for u in re.findall(r"https?://\S+", text, flags=re.IGNORECASE):
        u = u.strip().rstrip(").,;\"'")
        if u not in seen:
            seen.add(u)
            out.append(u)
    if not out:
        raise ValueError(f"No http(s) URLs in {path}")
    return out


def submit_triplet_screen_urls(
    base_url: str,
    target_text: str,
    urls: list[str],
    timeout: int,
    *,
    shuffle_seed: int | None,
    group_size: int,
) -> str:
    data: dict[str, str] = {
        "target_text": target_text,
        "urls_json": json.dumps(urls, ensure_ascii=False),
        "group_size": str(int(group_size)),
    }
    if shuffle_seed is not None:
        data["shuffle_seed"] = str(int(shuffle_seed))
    post_url = f"{base_url.rstrip('/')}/jobs/triplet_screen"
    try:
        resp = requests.post(post_url, data=data, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"POST {post_url} failed: {exc}\n"
            "Hint: ensure the rank API is reachable. Use --base-url or set SPEECHJUDGE_SMOKE_BASE_URL. "
            "Temporary *.trycloudflare.com hostnames fail DNS after the tunnel stops."
        ) from exc
    if resp.status_code >= 400:
        raise RuntimeError(f"POST failed: HTTP {resp.status_code}, body={resp.text[:2000]}")
    payload = resp.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"missing job_id: {payload}")
    return str(job_id)


def poll_job(base_url: str, job_id: str, timeout_seconds: int, interval: float) -> dict:
    deadline = time.time() + timeout_seconds
    url = f"{base_url.rstrip('/')}/jobs/{job_id}"
    last: dict = {}
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"GET {url} failed: {exc}\n"
                "Hint: check --base-url / SPEECHJUDGE_SMOKE_BASE_URL and that the tunnel or server is up."
            ) from exc
        last = resp.json()
        status = str(last.get("status", ""))
        phase = str(last.get("phase", ""))
        print(f"[poll] status={status} phase={phase}", flush=True)
        if status in {"succeeded", "failed"}:
            return last
        time.sleep(interval)
    raise TimeoutError(f"timeout; last={last}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke POST /jobs/triplet_screen")
    parser.add_argument(
        "--base-url",
        default=SMOKE_DEFAULT_BASE_URL,
        help="Rank API base URL (default: http://127.0.0.1:8010/ or SPEECHJUDGE_SMOKE_BASE_URL)",
    )
    parser.add_argument("--urls-file", type=Path, required=True)
    parser.add_argument("--target-text", default="smoke triplet screen")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="omit for random server seed")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--submit-timeout", type=int, default=1800)
    parser.add_argument("--job-timeout", type=int, default=28800)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    base_url = str(args.base_url).rstrip("/")

    urls = load_urls_from_file(args.urls_file)
    print(f"[info] base_url={base_url}", flush=True)
    print(f"[info] {len(urls)} URLs from {args.urls_file}", flush=True)

    job_id = submit_triplet_screen_urls(
        base_url,
        args.target_text,
        urls,
        args.submit_timeout,
        shuffle_seed=args.shuffle_seed,
        group_size=args.group_size,
    )
    print(f"[info] job_id={job_id}", flush=True)
    print(f"[info] job_url={base_url}/jobs/{job_id}", flush=True)
    result = poll_job(base_url, job_id, args.job_timeout, args.poll_interval)

    if str(result.get("status")) == "failed":
        err = result.get("error")
        msg = result.get("message")
        print("[error] job failed", flush=True)
        if msg:
            print(f"  message: {msg}", flush=True)
        if err:
            print(f"  error: {err}", flush=True)

    winners = result.get("winners") or []
    eliminated = result.get("eliminated") or []
    print("[summary]", flush=True)
    print(f"  winners: {len(winners)}", flush=True)
    for w in winners:
        print(
            f"    group {w.get('group_index')} win {w.get('wins_in_group')} id={w.get('id')}",
            flush=True,
        )
    print(f"  eliminated: {len(eliminated)}", flush=True)
    for e in eliminated:
        print(
            f"    group {e.get('group_index')} wins={e.get('wins_in_group')} "
            f"lost_to={e.get('lost_to_winner_id')} id={e.get('id')}",
            flush=True,
        )

    args.result_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.result_dir / f"{job_id}_triplet_screen.json"
    result_out = {
        "base_url": base_url,
        "job_url": f"{base_url}/jobs/{job_id}",
        **result,
    }
    out_path.write_text(json.dumps(result_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] saved {out_path}", flush=True)

    return 0 if str(result.get("status")) == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
