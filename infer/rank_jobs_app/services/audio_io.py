from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests


def download_url_to_file(url: str, dest_dir: Path, stem: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("audio_url must start with http:// or https://")

    suffix = Path(parsed.path).suffix or ".bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{stem}{suffix}"

    headers = {"User-Agent": "SpeechJudge-RankJobs/1.0"}
    connect_s, read_s = 30, 180
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


def ensure_wav(src_path: Path) -> Path:
    if src_path.suffix.lower() == ".wav":
        return src_path

    try:
        import librosa  # type: ignore[import-not-found]
        import soundfile as sf  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing librosa/soundfile (needed to convert non-wav inputs). "
            "Install: pip install librosa soundfile"
        ) from exc

    audio, sample_rate = librosa.load(str(src_path), sr=None, mono=False)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.T

    # Named temp next to infer conventions: caller may delete later.
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = Path(tmp.name)
    sf.write(str(wav_path), audio, sample_rate, format="WAV", subtype="PCM_16")
    return wav_path
