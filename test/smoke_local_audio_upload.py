"""Upload-test rank_jobs_app: by default **all** audios under a directory (full run).

Edit the ``SMOKE_*`` defaults below (or use CLI flags) before a run. Rank overrides are sent as
form fields ``rank_algorithm`` and ``full_pairwise_aggregation`` (server must implement current
``/jobs/rank``). Use value ``server`` for either flag to omit that field and rely on host env.

Examples::

    python test/smoke_local_audio_upload.py --base-url https://....trycloudflare.com
    python test/smoke_local_audio_upload.py --sample-size 3
    python test/smoke_local_audio_upload.py --urls-file test/date.txt --base-url https://....trycloudflare.com
    python test/smoke_local_audio_upload.py --urls-file test/date.txt --pair-votes-ramp 1,3
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import requests


# ---------------------------------------------------------------------------
# 联调默认：改这里即可；CLI 参数会覆盖对应项。
# ---------------------------------------------------------------------------
SMOKE_DEFAULT_BASE_URL = "https://bottles-possess-moss-austin.trycloudflare.com/"
SMOKE_DEFAULT_DATA_DIR = r"D:\Downloads\泰语"
SMOKE_DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "results"
SMOKE_DEFAULT_TARGET_TEXT = (
    "ตั้งแต่อายุยังน้อย คิโยซากิและไมค์ เพื่อนของเขามีความปรารถนาอย่างแรงกล้าที่จะกลายเป็นคนร่ำรวย อย่างไรก็ตาม ในตอนแรกพวกเขาไม่รู้ว่าจะทำอย่างไรจึงจะบรรลุเป้าหมายนี้ได้ เมื่อพวกเขาไปขอคำแนะนำจากพ่อของตนเอง พวกเขากลับได้รับคำตอบที่แตกต่างกันอย่างสิ้นเชิง พ่อที่ยากจนของคิโยซากิซึ่งมีการศึกษาดีแต่มีปัญหาทางการเงิน แนะนำให้พวกเขาตั้งใจเรียนและหางานที่มั่นคงทำ แม้คำแนะนำแบบดั้งเดิมนี้จะมาจากความหวังดี แต่มันมักทำให้ผู้คนติดอยู่ในวงจรของการทำงานหนักเพื่อเงิน โดยไม่สามารถสร้างความมั่งคั่งที่แท้จริงได้พ่อที่ยากจนของคิโยซากิเป็นตัวแทนของแนวคิดแบบดั้งเดิมที่ผู้คนจำนวนมากยังคงยึดถือมาจนถึงทุกวันนี้ แนวคิดนี้มักเกิดจากความกลัวต่อความไม่มั่นคงทางการเงิน และความเชื่อว่าการมีการศึกษาที่ดีและงานที่มั่นคงคือกุญแจสู่ความสำเร็จ อย่างไรก็ตาม คิโยซากิอธิบายว่าแนวทางนี้อาจทำให้ผู้คนติดอยู่ในสิ่งที่เรียกว่า “วงจรหนูวิ่ง” หรือ rat race ซึ่งหมายถึงการทำงานอย่างหนักเพื่อรับเงินเดือน แต่เงินจำนวนมากกลับถูกใช้ไปกับภาษี ค่าบิล และค่าใช้จ่ายต่าง ๆ ในชีวิตประจำวัน ดังนั้น แม้ว่าพวกเขาอาจหลีกเลี่ยงความยากจนได้ แต่ก็ยังไม่สามารถสะสมความมั่งคั่งที่แท้จริงได้เลย"
)
SMOKE_DEFAULT_SAMPLE_SIZE: int | None = None
SMOKE_DEFAULT_SEED = 42
SMOKE_DEFAULT_SUBMIT_TIMEOUT = 1800
SMOKE_DEFAULT_JOB_TIMEOUT = 28800
SMOKE_DEFAULT_POLL_INTERVAL = 5.0
SMOKE_DEFAULT_UPLOAD_HEARTBEAT_SEC = 30.0
SMOKE_DEFAULT_PAIRWISE_PARALLEL: int | None = None
SMOKE_DEFAULT_PAIR_VOTES_PER_PAIR = 1
# POST /jobs/rank：full_pairwise + bradley_terry 做最终排序；设为 "server" 则不传该字段（用服务端环境变量）
SMOKE_DEFAULT_RANK_ALGORITHM = "full_pairwise"
SMOKE_DEFAULT_FULL_PAIRWISE_AGGREGATION = "bradley_terry"

# 兼容旧常量名（供 argparse default 使用）
DEFAULT_BASE_URL = SMOKE_DEFAULT_BASE_URL
DEFAULT_DATA_DIR = SMOKE_DEFAULT_DATA_DIR
DEFAULT_RESULT_DIR = SMOKE_DEFAULT_RESULT_DIR
DEFAULT_TARGET_TEXT = SMOKE_DEFAULT_TARGET_TEXT

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
    pairwise_votes_per_pair: int | None = None,
    rank_algorithm: str | None = None,
    full_pairwise_aggregation: str | None = None,
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
    if pairwise_votes_per_pair is not None:
        data["pairwise_votes_per_pair"] = str(int(pairwise_votes_per_pair))
    if rank_algorithm is not None:
        data["rank_algorithm"] = str(rank_algorithm)
    if full_pairwise_aggregation is not None:
        data["full_pairwise_aggregation"] = str(full_pairwise_aggregation)
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
    pairwise_votes_per_pair: int | None = None,
    rank_algorithm: str | None = None,
    full_pairwise_aggregation: str | None = None,
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
        if pairwise_votes_per_pair is not None:
            post_data["pairwise_votes_per_pair"] = str(int(pairwise_votes_per_pair))
        if rank_algorithm is not None:
            post_data["rank_algorithm"] = str(rank_algorithm)
        if full_pairwise_aggregation is not None:
            post_data["full_pairwise_aggregation"] = str(full_pairwise_aggregation)
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
        done_c = last.get("comparisons_done")
        tot_c = last.get("comparisons_total")
        cmp_part = ""
        if isinstance(done_c, int) and isinstance(tot_c, int) and tot_c > 0:
            cmp_part = f" cmp={done_c}/{tot_c}"
        print(f"[poll] status={status} phase={phase} progress={progress:.3f}{cmp_part}", flush=True)
        if status in {"succeeded", "failed"}:
            return last
        time.sleep(interval)
    raise TimeoutError(f"job did not finish in {timeout_seconds}s; last={json.dumps(last, ensure_ascii=False)}")


def save_result_json(result_dir: Path, job_id: str, result: dict) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"{job_id}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_pair_votes_ramp(raw: str | None) -> list[int] | None:
    """Parse ``'1,3'`` or ``'1，3'`` (fullwidth comma) into ``[1, 3]``. Empty/None → None. Each entry must be 1 or 3 (server API)."""
    if raw is None:
        return None
    s = str(raw).strip()
    # IME / copy-paste: fullwidth comma U+FF0C, ideographic comma U+3001
    s = s.replace("\uff0c", ",").replace("\u3001", ",")
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = int(p)
        except ValueError as exc:
            raise ValueError(f"invalid integer in --pair-votes-ramp: {p!r}") from exc
        if v not in {1, 3}:
            raise ValueError(f"--pair-votes-ramp entries must be 1 or 3, got {v}")
        out.append(v)
    return out or None


def estimate_pairwise_desc(logical_pairs: int, pair_votes: int) -> str:
    if int(pair_votes) <= 1:
        return f"exact={logical_pairs} pairs x 1 vote = {logical_pairs} model comparisons on server"
    return (
        f"adaptive 2-of-3 voting, exact={logical_pairs} logical pairs, "
        f"model comparisons range={logical_pairs * 2}-{logical_pairs * 3}"
    )


def annotate_result_timing(result: dict) -> dict:
    created_at = _parse_dt(result.get("created_at"))
    started_at = _parse_dt(result.get("started_at"))
    finished_at = _parse_dt(result.get("finished_at"))

    timing: dict[str, float] = {}
    if created_at is not None and finished_at is not None:
        timing["queue_to_finish_seconds"] = round((finished_at - created_at).total_seconds(), 3)
    if started_at is not None and finished_at is not None:
        timing["run_seconds"] = round((finished_at - started_at).total_seconds(), 3)
    if timing:
        result["timing_summary"] = timing
    return result


def main() -> int:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(
        description=(
            "Rank job smoke test: either read http(s) URLs from --urls-file (small POST, tunnel-friendly), "
            "or upload local files from --data-dir."
        ),
    )
    parser.add_argument("--base-url", default=SMOKE_DEFAULT_BASE_URL, help="Service base URL")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=SMOKE_DEFAULT_RESULT_DIR,
        help="Directory where the final polled job JSON will be saved locally",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=None,
        help="Text file with one http(s) URL per line; submits urls_json (server downloads). E.g. test/date.txt",
    )
    parser.add_argument(
        "--data-dir", default=SMOKE_DEFAULT_DATA_DIR, help="Directory to recursively scan (ignored if --urls-file)"
    )
    parser.add_argument("--target-text", default=SMOKE_DEFAULT_TARGET_TEXT, help="Same transcript for all selected audios")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SMOKE_DEFAULT_SAMPLE_SIZE,
        metavar="N",
        help="If set, randomly sample N files; default uploads every audio found",
    )
    parser.add_argument("--seed", type=int, default=SMOKE_DEFAULT_SEED, help="Random seed when --sample-size is used")
    parser.add_argument(
        "--submit-timeout",
        type=int,
        default=SMOKE_DEFAULT_SUBMIT_TIMEOUT,
        help="Timeout seconds for POST /jobs/rank (many files need more)",
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=SMOKE_DEFAULT_JOB_TIMEOUT,
        help="Timeout seconds while polling (server default is now full pairwise ranking)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=SMOKE_DEFAULT_POLL_INTERVAL, help="Polling interval seconds"
    )
    parser.add_argument(
        "--upload-heartbeat-sec",
        type=float,
        default=SMOKE_DEFAULT_UPLOAD_HEARTBEAT_SEC,
        help="Print a line every N seconds while POST body is uploading (no per-chunk API in requests)",
    )
    parser.add_argument(
        "--pairwise-parallel",
        type=int,
        default=SMOKE_DEFAULT_PAIRWISE_PARALLEL,
        metavar="N",
        help="POST form pairwise_parallel (1–32): max pairs per batched GPU forward; omit for server default",
    )
    parser.add_argument(
        "--pair-votes-per-pair",
        type=int,
        default=SMOKE_DEFAULT_PAIR_VOTES_PER_PAIR,
        metavar="N",
        help="POST form pairwise_votes_per_pair (1 or 3): 1 = one model call per pair; 3 = adaptive 2-of-3; default=1",
    )
    parser.add_argument(
        "--pair-votes-ramp",
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated votes per pair (each 1 or 3): run that many separate rank jobs in order on the same "
            "URLs or files, e.g. 1,3 = one full job at 1 vote/pair then another at 2-of-3. Use ASCII comma or "
            "fullwidth ， between numbers. If set, each pass uses this sequence instead of --pair-votes-per-pair."
        ),
    )
    parser.add_argument(
        "--rank-algorithm",
        default=SMOKE_DEFAULT_RANK_ALGORITHM,
        choices=("full_pairwise", "phased_elo", "server"),
        help="POST rank_algorithm; use 'server' to omit and use host SPEECHJUDGE_RANK_ALGORITHM",
    )
    parser.add_argument(
        "--full-pairwise-aggregation",
        default=SMOKE_DEFAULT_FULL_PAIRWISE_AGGREGATION,
        choices=("round_robin_points", "bradley_terry", "bt", "rank_centrality_bt", "rc_bt", "server"),
        help="POST full_pairwise_aggregation (full_pairwise jobs); 'server' omits field for host env default",
    )
    args = parser.parse_args()
    if args.pairwise_parallel is not None and not (1 <= int(args.pairwise_parallel) <= 32):
        parser.error("--pairwise-parallel must be between 1 and 32")
    try:
        votes_ramp = parse_pair_votes_ramp(getattr(args, "pair_votes_ramp", None))
    except ValueError as exc:
        parser.error(str(exc))
    if votes_ramp is not None:
        votes_sequence = votes_ramp
    else:
        if int(args.pair_votes_per_pair) not in {1, 3}:
            parser.error("--pair-votes-per-pair must be either 1 or 3")
        votes_sequence = [int(args.pair_votes_per_pair)]

    post_rank_algorithm = None if args.rank_algorithm == "server" else str(args.rank_algorithm)
    post_full_pairwise_aggregation = (
        None if args.full_pairwise_aggregation == "server" else str(args.full_pairwise_aggregation)
    )
    print(
        f"[info] POST overrides: rank_algorithm={post_rank_algorithm or '(omit→server env)'} "
        f"full_pairwise_aggregation={post_full_pairwise_aggregation or '(omit→server env)'}",
        flush=True,
    )

    urls: list[str] | None = None
    chosen: list[Path] | None = None
    mode_label = ""

    if args.urls_file is not None:
        try:
            urls = load_urls_from_file(Path(args.urls_file))
        except (OSError, ValueError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        n = len(urls)
        logical_pairs = n * (n - 1) // 2 if n > 1 else 0
        first_votes = votes_sequence[0]
        est_desc = estimate_pairwise_desc(logical_pairs, first_votes)
        print(
            f"[info] mode=urls-file ({args.urls_file}), {n} URLs, "
            f"full pairwise mode, {est_desc}",
            flush=True,
        )
        if len(votes_sequence) > 1:
            print(
                f"[info] multi-pass --pair-votes-ramp: {votes_sequence} "
                f"({len(votes_sequence)} separate rank jobs on the same URLs)",
                flush=True,
            )
        if n > 40:
            print(
                (
                    "[info] full pairwise mode compares every unique pair once, so runtime grows quadratically "
                    "with the number of audios. Runtime is still dominated by model call cost."
                    if all(int(v) <= 1 for v in votes_sequence)
                    else "[info] adaptive 2-of-3 voting always spends 2 votes per pair and only adds the 3rd vote when the first two disagree."
                ),
                flush=True,
            )
        if n == 0:
            print("[error] no URLs loaded (need lines starting with http:// or https://)", file=sys.stderr)
            return 1
        for i, u in enumerate(urls[:15], start=1):
            print(f"  {i}. {u}", flush=True)
        if len(urls) > 15:
            print(f"  ... ({len(urls) - 15} more)", flush=True)
    else:
        data_dir = Path(args.data_dir)
        all_audios = collect_audios(data_dir)
        print(f"[info] found {len(all_audios)} audios under: {data_dir}", flush=True)
        if args.sample_size is None:
            chosen = sorted(all_audios, key=lambda p: str(p).lower())
            mode_label = "full (all files)"
        else:
            if args.sample_size < 1:
                parser.error("--sample-size must be >= 1")
            chosen = choose_random(all_audios, args.sample_size, args.seed)
            mode_label = f"sample n={args.sample_size}"
        n = len(chosen)
        logical_pairs = n * (n - 1) // 2 if n > 1 else 0
        first_votes = votes_sequence[0]
        est_desc = estimate_pairwise_desc(logical_pairs, first_votes)
        print(
            f"[info] mode={mode_label}, will upload {n} file(s), full pairwise mode, {est_desc}",
            flush=True,
        )
        if len(votes_sequence) > 1:
            print(
                f"[info] multi-pass --pair-votes-ramp: {votes_sequence} "
                f"({len(votes_sequence)} separate rank jobs on the same files)",
                flush=True,
            )
        if n == 0:
            print("[error] no audio files matched under --data-dir", file=sys.stderr)
            return 1
        print("[info] files:", flush=True)
        for i, p in enumerate(chosen, start=1):
            print(f"  {i}. {p}", flush=True)

    assert urls is not None or chosen is not None

    for pass_i, pair_votes in enumerate(votes_sequence, start=1):
        if len(votes_sequence) > 1:
            print(
                f"[info] pass {pass_i}/{len(votes_sequence)}: pairwise_votes_per_pair={pair_votes}",
                flush=True,
            )
        if urls is not None:
            job_id = submit_job_urls(
                args.base_url,
                args.target_text,
                urls,
                args.submit_timeout,
                pairwise_parallel=args.pairwise_parallel,
                pairwise_votes_per_pair=pair_votes,
                rank_algorithm=post_rank_algorithm,
                full_pairwise_aggregation=post_full_pairwise_aggregation,
            )
        else:
            assert chosen is not None
            job_id = submit_job(
                args.base_url,
                args.target_text,
                chosen,
                args.submit_timeout,
                upload_heartbeat_sec=args.upload_heartbeat_sec,
                pairwise_parallel=args.pairwise_parallel,
                pairwise_votes_per_pair=pair_votes,
                rank_algorithm=post_rank_algorithm,
                full_pairwise_aggregation=post_full_pairwise_aggregation,
            )
        print(f"[info] submitted job_id={job_id}", flush=True)
        result = annotate_result_timing(poll_job(args.base_url, job_id, args.job_timeout, args.poll_interval))

        print("[result]", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        timing = result.get("timing_summary")
        if isinstance(timing, dict):
            queue_to_finish = timing.get("queue_to_finish_seconds")
            run_seconds = timing.get("run_seconds")
            if isinstance(queue_to_finish, (int, float)):
                print(f"[info] total elapsed: {float(queue_to_finish):.3f}s", flush=True)
            if isinstance(run_seconds, (int, float)):
                print(f"[info] run elapsed: {float(run_seconds):.3f}s", flush=True)
        saved_path = save_result_json(Path(args.result_dir), job_id, result)
        print(f"[info] saved result json: {saved_path}", flush=True)
        if str(result.get("status")) != "succeeded":
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
