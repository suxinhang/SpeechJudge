"""Sequentially call a remote SpeechJudge API that downloads audio by URL.

Use this script when the remote service exposes ``/score-url`` and should
download + transcode + score the audio on the server side.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib import error, request


def check_health(base_url: str, timeout_s: float) -> dict:
    req = request.Request(f"{base_url.rstrip('/')}/health", method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_ready(
    base_url: str,
    timeout_s: float,
    poll_interval_s: float = 5.0,
) -> dict:
    start = time.time()
    while True:
        try:
            health = check_health(base_url, timeout_s=max(poll_interval_s, 5.0))
            if health.get("status") == "ready":
                return health
        except Exception as exc:
            _ = exc

        if timeout_s > 0 and (time.time() - start) >= timeout_s:
            raise TimeoutError(f"timed out after {timeout_s}s waiting for API ready")
        time.sleep(poll_interval_s)


def call_score_url(
    base_url: str,
    audio_url: str,
    target_text: str,
    timeout_s: float,
    max_new_tokens: int | None,
    mode: str | None = None,
    analysis: bool = False,
) -> dict:
    payload = {
        "audio_url": audio_url,
        "target_text": target_text,
        "analysis": analysis,
    }
    if mode:
        payload["mode"] = mode
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens

    req = request.Request(
        f"{base_url.rstrip('/')}/score-url",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iter_manifest_urls(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_url = row.get("url", "").strip()
            if audio_url:
                yield {
                    "base_name": row.get("base_name", ""),
                    "voice_name": row.get("voice_name", ""),
                    "gender": row.get("gender", ""),
                    "expected_score": row.get("score", ""),
                    "native_score": row.get("native_score", ""),
                    "audio_url": audio_url,
                }


def ensure_csv_header(csv_path: Path) -> None:
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "base_name",
                "voice_name",
                "gender",
                "expected_score",
                "native_score",
                "predicted_score",
                "naturalness",
                "accuracy",
                "emotion",
                "audio_url",
                "ok",
                "error",
                "elapsed_sec",
            ]
        )


def append_csv_row(csv_path: Path, row: list[str | int | float]) -> None:
    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()


def append_jsonl(jsonl_path: Path, obj: dict) -> None:
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score remote audio URLs through a SpeechJudge /score-url API."
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV with a url column.")
    parser.add_argument("--target", "-t", required=True, help="Expected transcript for all audio files.")
    parser.add_argument("--base-url", required=True, help="SpeechJudge API base URL.")
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=30.0,
        help="Total timeout for waiting API ready in seconds. Use 0 to wait forever.",
    )
    parser.add_argument(
        "--health-poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds while waiting for API ready.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=900.0,
        help="Timeout for each score request in seconds.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional decode cap passed to the API.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write result logs. Default: next to manifest.",
    )
    parser.add_argument(
        "--mode",
        default="compact",
        choices=["fast", "compact", "analysis"],
        help="Scoring mode sent to the API. Default: compact.",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Use analysis mode before extracting score. Slower but usually less collapsed.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    try:
        health = wait_for_ready(
            args.base_url,
            timeout_s=args.health_timeout,
            poll_interval_s=args.health_poll_interval,
        )
    except Exception as exc:
        raise SystemExit(f"API health check failed: {exc}") from exc

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "batch_score_analysis_results" if args.analysis else "batch_score_results"
    csv_path = output_dir / f"{prefix}.csv"
    jsonl_path = output_dir / f"{prefix}.jsonl"
    ensure_csv_header(csv_path)

    rows = list(iter_manifest_urls(manifest_path))
    total = len(rows)
    run_mode = "analysis" if args.analysis else args.mode
    print(f"API ready. URL scoring {total} file(s)... mode={run_mode}", flush=True)
    print(f"CSV log:   {csv_path}", flush=True)
    print(f"JSONL log: {jsonl_path}", flush=True)

    ok_count = 0
    for idx, item in enumerate(rows, start=1):
        start = time.time()
        try:
            response = call_score_url(
                args.base_url,
                audio_url=item["audio_url"],
                target_text=args.target,
                timeout_s=args.request_timeout,
                max_new_tokens=args.max_new_tokens,
                mode=run_mode,
                analysis=args.analysis,
            )
            predicted = response.get("score")
            sub_scores = response.get("sub_scores") or {}
            elapsed = round(time.time() - start, 2)
            ok = predicted is not None
            err = ""
            if ok:
                ok_count += 1
            log_obj = {
                "index": idx,
                "base_name": item["base_name"],
                "voice_name": item["voice_name"],
                "gender": item["gender"],
                "expected_score": item["expected_score"],
                "native_score": item["native_score"],
                "predicted_score": predicted,
                "audio_url": item["audio_url"],
                "ok": ok,
                "error": err,
                "elapsed_sec": elapsed,
                "api_response": response,
            }
            append_jsonl(jsonl_path, log_obj)
            append_csv_row(
                csv_path,
                [
                    idx,
                    item["base_name"],
                    item["voice_name"],
                    item["gender"],
                    item["expected_score"],
                    item["native_score"],
                    predicted,
                    sub_scores.get("naturalness", ""),
                    sub_scores.get("accuracy", ""),
                    sub_scores.get("emotion", ""),
                    item["audio_url"],
                    ok,
                    err,
                    elapsed,
                ],
            )
            print(f"[{idx}/{total}] {item['base_name']} -> {predicted} (elapsed {elapsed}s)", flush=True)
        except error.HTTPError as exc:
            elapsed = round(time.time() - start, 2)
            detail = exc.read().decode("utf-8", errors="replace")
            log_obj = {
                "index": idx,
                "base_name": item["base_name"],
                "voice_name": item["voice_name"],
                "gender": item["gender"],
                "expected_score": item["expected_score"],
                "native_score": item["native_score"],
                "predicted_score": None,
                "audio_url": item["audio_url"],
                "ok": False,
                "error": f"HTTP {exc.code}: {detail}",
                "elapsed_sec": elapsed,
            }
            append_jsonl(jsonl_path, log_obj)
            append_csv_row(
                csv_path,
                [
                    idx,
                    item["base_name"],
                    item["voice_name"],
                    item["gender"],
                    item["expected_score"],
                    item["native_score"],
                    "",
                    "",
                    "",
                    "",
                    item["audio_url"],
                    False,
                    f"HTTP {exc.code}: {detail}",
                    elapsed,
                ],
            )
            print(f"[{idx}/{total}] {item['base_name']} -> ERROR HTTP {exc.code} (elapsed {elapsed}s)", flush=True)
        except Exception as exc:
            elapsed = round(time.time() - start, 2)
            log_obj = {
                "index": idx,
                "base_name": item["base_name"],
                "voice_name": item["voice_name"],
                "gender": item["gender"],
                "expected_score": item["expected_score"],
                "native_score": item["native_score"],
                "predicted_score": None,
                "audio_url": item["audio_url"],
                "ok": False,
                "error": str(exc),
                "elapsed_sec": elapsed,
            }
            append_jsonl(jsonl_path, log_obj)
            append_csv_row(
                csv_path,
                [
                    idx,
                    item["base_name"],
                    item["voice_name"],
                    item["gender"],
                    item["expected_score"],
                    item["native_score"],
                    "",
                    "",
                    "",
                    "",
                    item["audio_url"],
                    False,
                    str(exc),
                    elapsed,
                ],
            )
            print(f"[{idx}/{total}] {item['base_name']} -> ERROR {exc} (elapsed {elapsed}s)", flush=True)

    print(f"Done. Successful results: {ok_count}/{total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
