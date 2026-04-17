"""Upload-test rank_jobs_app: by default **all** audios under a directory (full run).

Default data directory:
    D:\\Downloads\\泰语

Use ``--sample-size N`` for a quick random subset only.

Example::

    python test/smoke_local_audio_upload.py --base-url https://....trycloudflare.com
    python test/smoke_local_audio_upload.py --sample-size 3
    python test/smoke_local_audio_upload.py --urls-file test/date.txt --base-url https://....trycloudflare.com
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import requests


DEFAULT_BASE_URL = "https://madonna-perspectives-ctrl-illustration.trycloudflare.com"
DEFAULT_DATA_DIR = r"D:\Downloads\泰语"
DEFAULT_TARGET_TEXT = "ตั้งแต่อายุยังน้อย คิโยซากิและไมค์ เพื่อนของเขามีความปรารถนาอย่างแรงกล้าที่จะกลายเป็นคนร่ำรวย อย่างไรก็ตาม ในตอนแรกพวกเขาไม่รู้ว่าจะทำอย่างไรจึงจะบรรลุเป้าหมายนี้ได้ เมื่อพวกเขาไปขอคำแนะนำจากพ่อของตนเอง พวกเขากลับได้รับคำตอบที่แตกต่างกันอย่างสิ้นเชิง พ่อที่ยากจนของคิโยซากิซึ่งมีการศึกษาดีแต่มีปัญหาทางการเงิน แนะนำให้พวกเขาตั้งใจเรียนและหางานที่มั่นคงทำ แม้คำแนะนำแบบดั้งเดิมนี้จะมาจากความหวังดี แต่มันมักทำให้ผู้คนติดอยู่ในวงจรของการทำงานหนักเพื่อเงิน โดยไม่สามารถสร้างความมั่งคั่งที่แท้จริงได้พ่อที่ยากจนของคิโยซากิเป็นตัวแทนของแนวคิดแบบดั้งเดิมที่ผู้คนจำนวนมากยังคงยึดถือมาจนถึงทุกวันนี้ แนวคิดนี้มักเกิดจากความกลัวต่อความไม่มั่นคงทางการเงิน และความเชื่อว่าการมีการศึกษาที่ดีและงานที่มั่นคงคือกุญแจสู่ความสำเร็จ อย่างไรก็ตาม คิโยซากิอธิบายว่าแนวทางนี้อาจทำให้ผู้คนติดอยู่ในสิ่งที่เรียกว่า “วงจรหนูวิ่ง” หรือ rat race ซึ่งหมายถึงการทำงานอย่างหนักเพื่อรับเงินเดือน แต่เงินจำนวนมากกลับถูกใช้ไปกับภาษี ค่าบิล และค่าใช้จ่ายต่าง ๆ ในชีวิตประจำวัน ดังนั้น แม้ว่าพวกเขาอาจหลีกเลี่ยงความยากจนได้ แต่ก็ยังไม่สามารถสะสมความมั่งคั่งที่แท้จริงได้เลย"

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}


def _configure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def collect_audios(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"audio directory not found: {root}")
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return out


def choose_random(paths: Iterable[Path], n: int, seed: int) -> list[Path]:
    items = list(paths)
    if len(items) < n:
        raise ValueError(f"need at least {n} audio files, got {len(items)}")
    rng = random.Random(seed)
    return rng.sample(items, n)


def load_urls_from_file(path: Path) -> list[str]:
    """Load http(s) URLs from a text file.

    Uses regex over the whole file so it still works if URLs are on **one long line**
    (no newlines), or if a UTF-8 BOM would break line-based ``startswith`` checks.
    """
    if not path.is_file():
        raise FileNotFoundError(f"URL list file not found: {path.resolve()}")
    sz = path.stat().st_size
    if sz == 0:
        raise ValueError(
            f"{path.resolve()} is empty on disk (0 bytes). "
            "If your editor tab shows URLs, save the file (Ctrl+S) and run again."
        )
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    seen: set[str] = set()
    out: list[str] = []
    for u in re.findall(r"https?://\S+", text, flags=re.IGNORECASE):
        u = u.strip().rstrip(").,;\"'")
        if u not in seen:
            seen.add(u)
            out.append(u)
    if not out:
        preview = text[:300].replace("\r", "").replace("\n", "\\n")
        raise ValueError(
            f"No http(s) URLs matched in {path.resolve()} ({sz} bytes). Preview: {preview!r}"
        )
    return out


def submit_job_urls(
    base_url: str,
    target_text: str,
    urls: list[str],
    timeout: int,
    *,
    pairwise_parallel: int | None = None,
) -> str:
    body = json.dumps(urls, ensure_ascii=False)
    print(
        f"[info] POST /jobs/rank with urls_json only ({len(urls)} URLs, ~{len(body)} bytes form payload)…",
        flush=True,
    )
    data: dict[str, str] = {
        "target_text": target_text,
        "urls_json": body,
    }
    if pairwise_parallel is not None:
        data["pairwise_parallel"] = str(int(pairwise_parallel))
    resp = requests.post(
        f"{base_url.rstrip('/')}/jobs/rank",
        data=data,
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"POST /jobs/rank failed: HTTP {resp.status_code}, body={resp.text[:2000]}")
    payload = resp.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"missing job_id in response: {json.dumps(payload, ensure_ascii=False)}")
    return str(job_id)


def submit_job(
    base_url: str,
    target_text: str,
    files: list[Path],
    timeout: int,
    *,
    upload_heartbeat_sec: float = 30.0,
    pairwise_parallel: int | None = None,
) -> str:
    total_bytes = sum(p.stat().st_size for p in files)
    mib = total_bytes / (1024 * 1024)
    print(
        f"[info] POST /jobs/rank: {len(files)} files, ~{mib:.1f} MiB — "
        "multipart upload runs inside requests until the full body is sent "
        f"(heartbeat every {upload_heartbeat_sec:.0f}s while uploading)…",
        flush=True,
    )

    multipart = []
    opened = []
    try:
        for i, p in enumerate(files):
            f = p.open("rb")
            opened.append(f)
            upload_name = f"{i:04d}_{p.name}"
            multipart.append(("audio_files", (upload_name, f, "application/octet-stream")))

        stop_hb = threading.Event()

        def _heartbeat() -> None:
            tick = 0
            interval = max(5.0, float(upload_heartbeat_sec))
            while not stop_hb.wait(interval):
                tick += 1
                elapsed = int(tick * interval)
                print(f"[info] still uploading multipart… ~{elapsed}s elapsed", flush=True)

        hb = threading.Thread(target=_heartbeat, name="upload-heartbeat", daemon=True)
        hb.start()
        post_data: dict[str, str] = {"target_text": target_text}
        if pairwise_parallel is not None:
            post_data["pairwise_parallel"] = str(int(pairwise_parallel))
        try:
            resp = requests.post(
                f"{base_url.rstrip('/')}/jobs/rank",
                data=post_data,
                files=multipart,
                timeout=timeout,
            )
        finally:
            stop_hb.set()
            hb.join(timeout=2.0)
    finally:
        for f in opened:
            f.close()

    print("[info] upload finished, server responded", flush=True)

    if resp.status_code >= 400:
        body = resp.text
        raise RuntimeError(f"POST /jobs/rank failed: HTTP {resp.status_code}, body={body}")
    payload = resp.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"missing job_id in response: {json.dumps(payload, ensure_ascii=False)}")
    return str(job_id)


def poll_job(base_url: str, job_id: str, timeout_seconds: int, interval: float) -> dict:
    deadline = time.time() + timeout_seconds
    url = f"{base_url.rstrip('/')}/jobs/{job_id}"
    last = {}
    print(f"[info] polling GET {url} every {interval}s…", flush=True)
    while time.time() < deadline:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        last = resp.json()
        status = str(last.get("status", ""))
        phase = str(last.get("phase", ""))
        progress = float(last.get("progress", 0.0) or 0.0)
        print(f"[poll] status={status} phase={phase} progress={progress:.3f}", flush=True)
        if status in {"succeeded", "failed"}:
            return last
        time.sleep(interval)
    raise TimeoutError(f"job did not finish in {timeout_seconds}s; last={json.dumps(last, ensure_ascii=False)}")


def main() -> int:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(
        description=(
            "Rank job smoke test: either read http(s) URLs from --urls-file (small POST, tunnel-friendly), "
            "or upload local files from --data-dir."
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Service base URL")
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=None,
        help="Text file with one http(s) URL per line; submits urls_json (server downloads). E.g. test/date.txt",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory to recursively scan (ignored if --urls-file)")
    parser.add_argument("--target-text", default=DEFAULT_TARGET_TEXT, help="Same transcript for all selected audios")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help="If set, randomly sample N files; default uploads every audio found",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed when --sample-size is used")
    parser.add_argument(
        "--submit-timeout",
        type=int,
        default=1800,
        help="Timeout seconds for POST /jobs/rank (many files need more)",
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=28800,
        help="Timeout seconds while polling (bubble sort ~ n*(n-1)/2 comparisons)",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval seconds")
    parser.add_argument(
        "--upload-heartbeat-sec",
        type=float,
        default=30.0,
        help="Print a line every N seconds while POST body is uploading (no per-chunk API in requests)",
    )
    parser.add_argument(
        "--pairwise-parallel",
        type=int,
        default=None,
        metavar="N",
        help="POST form pairwise_parallel (1–32); omit to use server default",
    )
    args = parser.parse_args()
    if args.pairwise_parallel is not None and not (1 <= int(args.pairwise_parallel) <= 32):
        parser.error("--pairwise-parallel must be between 1 and 32")

    if args.urls_file is not None:
        try:
            urls = load_urls_from_file(Path(args.urls_file))
        except (OSError, ValueError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        n = len(urls)
        est = n * (n - 1) // 2 if n > 1 else 0
        print(f"[info] mode=urls-file ({args.urls_file}), {n} URLs, ~{est} pairwise comparisons on server", flush=True)
        if n == 0:
            print("[error] no URLs loaded (need lines starting with http:// or https://)", file=sys.stderr)
            return 1
        for i, u in enumerate(urls[:15], start=1):
            print(f"  {i}. {u}", flush=True)
        if len(urls) > 15:
            print(f"  ... ({len(urls) - 15} more)", flush=True)
        job_id = submit_job_urls(
            args.base_url,
            args.target_text,
            urls,
            args.submit_timeout,
            pairwise_parallel=args.pairwise_parallel,
        )
    else:
        data_dir = Path(args.data_dir)
        all_audios = collect_audios(data_dir)
        print(f"[info] found {len(all_audios)} audios under: {data_dir}", flush=True)
        if args.sample_size is None:
            chosen = sorted(all_audios, key=lambda p: str(p).lower())
            mode = "full (all files)"
        else:
            if args.sample_size < 1:
                parser.error("--sample-size must be >= 1")
            chosen = choose_random(all_audios, args.sample_size, args.seed)
            mode = f"sample n={args.sample_size}"
        n = len(chosen)
        est = n * (n - 1) // 2 if n > 1 else 0
        print(
            f"[info] mode={mode}, will upload {n} file(s), ~{est} pairwise comparisons on server",
            flush=True,
        )
        if n == 0:
            print("[error] no audio files matched under --data-dir", file=sys.stderr)
            return 1
        print("[info] files:", flush=True)
        for i, p in enumerate(chosen, start=1):
            print(f"  {i}. {p}", flush=True)

        job_id = submit_job(
            args.base_url,
            args.target_text,
            chosen,
            args.submit_timeout,
            upload_heartbeat_sec=args.upload_heartbeat_sec,
            pairwise_parallel=args.pairwise_parallel,
        )
    print(f"[info] submitted job_id={job_id}", flush=True)
    result = poll_job(args.base_url, job_id, args.job_timeout, args.poll_interval)

    print("[result]", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if str(result.get("status")) != "succeeded":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
