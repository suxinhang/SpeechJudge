"""Decode media files (MP4, MP3, …) to WAV for SpeechJudge inference.

MP4/MOV/MKV 等容器优先用 **ffmpeg** 只抽取音轨；不可用时回退 **librosa**。
服务器上建议安装 ffmpeg: ``apt install ffmpeg`` 或等价包。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def _ffmpeg_extract_wav(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-map",
        "0:a:0",
        "-f",
        "wav",
        str(dst),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(err[:2000])


def decode_to_wav(src_path: Path) -> Path:
    """Return ``src_path`` if already WAV; otherwise write a new temp ``.wav`` and return its path."""

    if not src_path.is_file():
        raise FileNotFoundError(str(src_path))
    if src_path.suffix.lower() == ".wav":
        return src_path

    errs: list[str] = []

    if shutil.which("ffmpeg"):
        tmp_ff = Path(tempfile.mkstemp(suffix=".wav", prefix="sj_ff_")[1])
        try:
            _ffmpeg_extract_wav(src_path, tmp_ff)
            return tmp_ff
        except Exception as exc:
            errs.append(f"ffmpeg: {exc!r}")
            tmp_ff.unlink(missing_ok=True)

    try:
        import librosa  # type: ignore[import-not-found]
        import soundfile as sf  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing librosa/soundfile (pip install librosa soundfile). "
            "For MP4/MOV/MKV install ffmpeg as well."
        ) from exc

    tmp_lr = Path(tempfile.mkstemp(suffix=".wav", prefix="sj_lr_")[1])
    try:
        audio, sample_rate = librosa.load(str(src_path), sr=None, mono=False)
        if getattr(audio, "ndim", 1) == 2:
            audio = audio.T
        sf.write(str(tmp_lr), audio, sample_rate, format="WAV", subtype="PCM_16")
        return tmp_lr
    except Exception as exc:
        errs.append(f"librosa: {exc!r}")
        tmp_lr.unlink(missing_ok=True)

    raise RuntimeError(
        f"Cannot decode audio from {src_path.name}: {'; '.join(errs)}. "
        "Install ffmpeg on the server to extract audio from MP4 and similar containers."
    )
