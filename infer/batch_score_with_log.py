"""Sequentially score WAV files through the local SpeechJudge API and log each result immediately.

Features:
- Reads files either from a directory or a manifest CSV
- Scores one file at a time
- Appends each completed result to CSV and JSONL immediately
- Prints one line per completed file with flush=True so progress is visible live

Example:
    python batch_score_with_log.py ^
      --wav-dir "D:\\work\\tts\\SpeechJudge\\infer\\examples\\user_demo\\drive_batch\\wav" ^
      --target "Your transcript here"

    python batch_score_with_log.py ^
      --manifest "D:\\work\\tts\\SpeechJudge\\infer\\examples\\user_demo\\drive_batch\\manifest.csv" ^
      --target "Your transcript here"
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
    """
    Wait until the API reports ``status=ready``.

    ``timeout_s <= 0`` means wait forever.
    """
    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            health = check_health(base_url, timeout_s=max(poll_interval_s, 5.0))
            status = health.get("status")
            if status == "ready":
                return health
        except Exception as exc:
            _ = exc

        if timeout_s > 0 and (time.time() - start) >= timeout_s:
            raise TimeoutError(f"timed out after {timeout_s}s waiting for API ready")
        time.sleep(poll_interval_s)


def call_score_path(
    base_url: str,
    audio_path: Path,
    target_text: str,
    timeout_s: float,
    max_new_tokens: int | None,
    mode: str | None = None,
    analysis: bool = False,
) -> dict:
    payload = {
        "audio_path": str(audio_path),
        "target_text": target_text,
        "analysis": analysis,
    }
    if mode:
        payload["mode"] = mode
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens

    req = request.Request(
        f"{base_url.rstrip('/')}/score-path",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iter_manifest_wavs(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = Path(row["wav_path"])
            if wav_path.is_file():
                yield {
                    "base_name": row.get("base_name") or wav_path.stem,
                    "voice_name": row.get("voice_name", ""),
                    "gender": row.get("gender", ""),
                    "expected_score": row.get("score", ""),
                    "wav_path": wav_path,
                }


def iter_wav_dir(wav_dir: Path):
    for wav_path in sorted(wav_dir.glob("*.wav")):
        yield {
            "base_name": wav_path.stem,
            "voice_name": "",
            "gender": "",
            "expected_score": "",
            "wav_path": wav_path,
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
                "predicted_score",
                "naturalness",
                "accuracy",
                "emotion",
                "wav_path",
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
        description="Score WAV files one by one and log each result immediately."
    )
    parser.add_argument(
        "--wav-dir",
        default=None,
        help="Directory containing WAV files to score.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest CSV with wav_path/base_name metadata.",
    )
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Expected transcript for all audio files.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="SpeechJudge API base URL.",
    )
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
        help="Directory to write result logs. Default: next to manifest or wav dir.",
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

    if not args.wav_dir and not args.manifest:
        raise SystemExit("Provide either --wav-dir or --manifest.")

    manifest_path = Path(args.manifest) if args.manifest else None
    wav_dir = Path(args.wav_dir) if args.wav_dir else None

    if manifest_path is not None and not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if wav_dir is not None and not wav_dir.is_dir():
        raise SystemExit(f"WAV directory not found: {wav_dir}")

    try:
        health = wait_for_ready(
            args.base_url,
            timeout_s=args.health_timeout,
            poll_interval_s=args.health_poll_interval,
        )
    except Exception as exc:
        raise SystemExit(f"API health check failed: {exc}") from exc

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif manifest_path is not None:
        output_dir = manifest_path.parent
    else:
        output_dir = wav_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "batch_score_analysis_results" if args.analysis else "batch_score_results"
    csv_path = output_dir / f"{prefix}.csv"
    jsonl_path = output_dir / f"{prefix}.jsonl"
    ensure_csv_header(csv_path)

    rows = list(iter_manifest_wavs(manifest_path)) if manifest_path else list(iter_wav_dir(wav_dir))
    total = len(rows)
    run_mode = "analysis" if args.analysis else args.mode
    print(
        f"API ready. Scoring {total} file(s)... mode={run_mode}",
        flush=True,
    )
    print(f"CSV log:   {csv_path}", flush=True)
    print(f"JSONL log: {jsonl_path}", flush=True)

    ok_count = 0
    for idx, item in enumerate(rows, start=1):
        wav_path = item["wav_path"]
        start = time.time()
        try:
            response = call_score_path(
                args.base_url,
                audio_path=wav_path,
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
                "predicted_score": predicted,
                "wav_path": str(wav_path),
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
                    predicted,
                    sub_scores.get("naturalness", ""),
                    sub_scores.get("accuracy", ""),
                    sub_scores.get("emotion", ""),
                    str(wav_path),
                    ok,
                    err,
                    elapsed,
                ],
            )
            print(
                f"[{idx}/{total}] {item['base_name']} -> {predicted} (elapsed {elapsed}s)",
                flush=True,
            )
        except error.HTTPError as exc:
            elapsed = round(time.time() - start, 2)
            detail = exc.read().decode("utf-8", errors="replace")
            log_obj = {
                "index": idx,
                "base_name": item["base_name"],
                "voice_name": item["voice_name"],
                "gender": item["gender"],
                "expected_score": item["expected_score"],
                "predicted_score": None,
                "wav_path": str(wav_path),
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
                    "",
                    "",
                    "",
                    "",
                    str(wav_path),
                    False,
                    f"HTTP {exc.code}: {detail}",
                    elapsed,
                ],
            )
            print(
                f"[{idx}/{total}] {item['base_name']} -> ERROR HTTP {exc.code} (elapsed {elapsed}s)",
                flush=True,
            )
        except Exception as exc:
            elapsed = round(time.time() - start, 2)
            log_obj = {
                "index": idx,
                "base_name": item["base_name"],
                "voice_name": item["voice_name"],
                "gender": item["gender"],
                "expected_score": item["expected_score"],
                "predicted_score": None,
                "wav_path": str(wav_path),
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
                    "",
                    "",
                    "",
                    "",
                    str(wav_path),
                    False,
                    str(exc),
                    elapsed,
                ],
            )
            print(
                f"[{idx}/{total}] {item['base_name']} -> ERROR {exc} (elapsed {elapsed}s)",
                flush=True,
            )

    print(f"Done. Successful results: {ok_count}/{total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
