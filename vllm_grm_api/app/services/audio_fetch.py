from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


def _suffix_from_url(url: str) -> str:
    path = urlparse(url).path
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        if path.lower().endswith(ext):
            return ext
    return ".bin"


def download_audio_to_cache(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    dest = cache_dir / f"{h}{_suffix_from_url(url)}"
    if dest.is_file():
        return dest

    last_err: Exception | None = None
    # S3 occasionally returns transient TLS EOF / connect reset; retry with backoff.
    for attempt in range(1, 6):
        try:
            with httpx.Client(
                timeout=120.0,
                follow_redirects=True,
                headers={"User-Agent": "SpeechJudge-vllm-grm-api/1.0"},
            ) as client:
                r = client.get(url)
                # Retry on transient server-side status only.
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"transient status={r.status_code}",
                        request=r.request,
                        response=r,
                    )
                r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return dest
        except (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
        ) as e:
            last_err = e
            if attempt == 5:
                break
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"failed to download audio after retries: {url}") from last_err
