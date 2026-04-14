"""Client helper for the persistent SpeechJudge API.

The script normalizes every input audio file into a temporary WAV file before
uploading it to ``/score-upload``. This avoids local-path assumptions on the
server side and keeps the model input format consistent.

Examples:
    python call_score_api.py ^
      "examples\\user_demo\\a.mp3" ^
      --target "Hello world."

    python call_score_api.py ^
      "a.mp3" "b.wav" ^
      --target "Hello world." ^
      --analysis ^
      --base-url "http://127.0.0.1:8000"
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import tempfile
import uuid
from pathlib import Path
from urllib import error, request

import librosa
import soundfile as sf


def convert_to_wav(src_path: Path) -> Path:
    """Decode any supported audio file and rewrite it as PCM WAV."""
    audio, sample_rate = librosa.load(str(src_path), sr=None, mono=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = Path(tmp.name)

    if getattr(audio, "ndim", 1) == 2:
        audio = audio.T
    sf.write(str(wav_path), audio, sample_rate, format="WAV", subtype="PCM_16")
    return wav_path


def build_multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----SpeechJudgeBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    mime_type = mimetypes.guess_type(file_path.name)[0] or "audio/wav"
    file_bytes = file_path.read_bytes()
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    return b"".join(chunks), boundary


def wait_until_ready(base_url: str, timeout_s: float) -> dict:
    req = request.Request(f"{base_url.rstrip('/')}/health", method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload


def score_one_audio(
    base_url: str,
    audio_path: Path,
    target_text: str,
    analysis: bool,
    max_new_tokens: int | None,
    timeout_s: float,
) -> dict:
    wav_path = convert_to_wav(audio_path)
    try:
        fields = {
            "target_text": target_text,
            "analysis": str(analysis).lower(),
        }
        if max_new_tokens is not None:
            fields["max_new_tokens"] = str(max_new_tokens)
        body, boundary = build_multipart_body(fields, "audio", wav_path)
        req = request.Request(
            f"{base_url.rstrip('/')}/score-upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        payload["source_audio_path"] = str(audio_path)
        payload["uploaded_wav_path"] = str(wav_path)
        return payload
    finally:
        wav_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert local audio to WAV and call the SpeechJudge API."
    )
    parser.add_argument("audio", nargs="+", help="One or more local audio files.")
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Expected transcript shared by the input audio(s).",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="SpeechJudge API base URL (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Request slow analysis mode instead of fast score-only mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional decode cap forwarded to the API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds for each request (default: 600).",
    )
    args = parser.parse_args()

    try:
        health = wait_until_ready(args.base_url, timeout_s=10.0)
    except error.URLError as exc:
        raise SystemExit(f"API unreachable: {exc}") from exc

    if health.get("status") != "ready":
        raise SystemExit(f"API not ready: {health}")

    results = []
    for audio_str in args.audio:
        audio_path = Path(audio_str)
        if not audio_path.is_file():
            raise SystemExit(f"Audio file not found: {audio_path}")
        try:
            result = score_one_audio(
                base_url=args.base_url,
                audio_path=audio_path,
                target_text=args.target,
                analysis=args.analysis,
                max_new_tokens=args.max_new_tokens,
                timeout_s=args.timeout,
            )
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"API error {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise SystemExit(f"API request failed: {exc}") from exc
        results.append(result)

    print(json.dumps({"health": health, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
