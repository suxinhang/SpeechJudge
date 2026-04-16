"""Smoke test rank_jobs_app by uploading 3 random local audios.

Default data directory:
    D:\\Downloads\\泰语

Example:
    python test/smoke_local_audio_upload.py \
      --base-url https://explore-psp-trustees-hanging.trycloudflare.com
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import requests


DEFAULT_BASE_URL = "https://since-supporting-edwards-comments.trycloudflare.com/"
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


def submit_job(base_url: str, target_text: str, files: list[Path], timeout: int) -> str:
    multipart = []
    opened = []
    try:
        for p in files:
            f = p.open("rb")
            opened.append(f)
            multipart.append(("audio_files", (p.name, f, "application/octet-stream")))

        resp = requests.post(
            f"{base_url.rstrip('/')}/jobs/rank",
            data={"target_text": target_text},
            files=multipart,
            timeout=timeout,
        )
    finally:
        for f in opened:
            f.close()

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
    while time.time() < deadline:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        last = resp.json()
        status = str(last.get("status", ""))
        phase = str(last.get("phase", ""))
        progress = float(last.get("progress", 0.0) or 0.0)
        print(f"[poll] status={status} phase={phase} progress={progress:.3f}")
        if status in {"succeeded", "failed"}:
            return last
        time.sleep(interval)
    raise TimeoutError(f"job did not finish in {timeout_seconds}s; last={json.dumps(last, ensure_ascii=False)}")


def main() -> int:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(description="Smoke test upload API with 3 random local audios.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Service base URL")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory to recursively scan")
    parser.add_argument("--target-text", default=DEFAULT_TARGET_TEXT, help="Same transcript for all selected audios")
    parser.add_argument("--sample-size", type=int, default=3, help="How many files to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--submit-timeout", type=int, default=120, help="Timeout seconds for POST /jobs/rank")
    parser.add_argument("--job-timeout", type=int, default=1800, help="Timeout seconds while polling job result")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval seconds")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_audios = collect_audios(data_dir)
    print(f"[info] found {len(all_audios)} audios under: {data_dir}")
    chosen = choose_random(all_audios, args.sample_size, args.seed)
    print("[info] sampled files:")
    for i, p in enumerate(chosen, start=1):
        print(f"  {i}. {p}")

    job_id = submit_job(args.base_url, args.target_text, chosen, args.submit_timeout)
    print(f"[info] submitted job_id={job_id}")
    result = poll_job(args.base_url, job_id, args.job_timeout, args.poll_interval)

    print("[result]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if str(result.get("status")) != "succeeded":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
