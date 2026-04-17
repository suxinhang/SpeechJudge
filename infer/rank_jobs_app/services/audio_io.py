from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests import exceptions as req_exc

# infer/ is on sys.path when rank_jobs_app starts
from audio_decode import decode_to_wav

# Transient TLS / TCP drops when pulling many objects from S3 (or any CDN).
_RETRYABLE_GET = (
    req_exc.SSLError,
    req_exc.ConnectionError,
    req_exc.ChunkedEncodingError,
    req_exc.Timeout,
)

# After raise_for_status: CDN / origin blips worth another round-trip.
_RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def download_url_to_file(url: str, dest_dir: Path, stem: str, *, max_attempts: int = 5) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("audio_url must start with http:// or https://")

    suffix = Path(parsed.path).suffix or ".bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{stem}{suffix}"

    headers = {"User-Agent": "SpeechJudge-RankJobs/1.0"}
    connect_s, read_s = 30, 300
    last_err: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(connect_s, read_s),
                headers=headers,
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return out_path
        except req_exc.HTTPError as exc:
            resp = exc.response
            code = resp.status_code if resp is not None else None
            if code in _RETRYABLE_HTTP_STATUS and attempt < max_attempts:
                last_err = exc
                wait = min(2**attempt, 30)
                if code == 429 and resp is not None:
                    ra = (resp.headers.get("Retry-After") or "").strip().split(",")[0]
                    if ra.isdigit():
                        wait = max(wait, min(int(ra), 120))
                time.sleep(wait)
                continue
            raise
        except _RETRYABLE_GET as exc:
            last_err = exc
            if attempt >= max_attempts:
                break
            wait = min(2**attempt, 30)
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def ensure_wav(src_path: Path, *, max_attempts: int = 3) -> Path:
    """Return a path to a WAV suitable for inference (MP4/MOV/… → ffmpeg + librosa fallback).

    Retries help with transient ffmpeg I/O or librosa read glitches on shared disks.
    """
    last_err: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return decode_to_wav(src_path)
        except FileNotFoundError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            last_err = exc
            if attempt >= max_attempts:
                break
            wait = min(2**attempt, 20)
            time.sleep(wait)
    assert last_err is not None
    raise last_err
