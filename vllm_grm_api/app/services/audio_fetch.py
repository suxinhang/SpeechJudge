from __future__ import annotations

import hashlib
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
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
    dest.write_bytes(r.content)
    return dest
