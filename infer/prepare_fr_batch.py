from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from urllib.request import urlretrieve

import librosa
import soundfile as sf


ITEMS = [
    {
        "voice_name": "fr-FR-CelesteNeural",
        "gender": "",
        "hexin_score": 8,
        "native_score": 8,
        "url": "https://overseas-resource-storage.s3.us-west-2.amazonaws.com/explore/tts_demo/audio/364_a44527d4.mp3",
    },
    {
        "voice_name": "fr-FR-ClaudeNeural",
        "gender": "",
        "hexin_score": 8,
        "native_score": 8,
        "url": "https://overseas-resource-storage.s3.us-west-2.amazonaws.com/explore/tts_demo/audio/365_13925258.mp3",
    },
    {
        "voice_name": "te",
        "gender": "",
        "hexin_score": 6,
        "native_score": 10,
        "url": "https://overseas-resource-storage.s3.us-west-2.amazonaws.com/explore/tts_demo/audio/356_4c4fc034.mp3",
    },
    {
        "voice_name": "fr-FR-Chirp3-HD-Leda",
        "gender": "",
        "hexin_score": 4,
        "native_score": 0,
        "url": "https://overseas-resource-storage.s3.us-west-2.amazonaws.com/explore/tts_demo/audio/395_a743a407.mp3",
    },
]


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "audio"


def convert_to_wav(src_path: Path, wav_dir: Path) -> tuple[Path, int]:
    audio, sample_rate = librosa.load(str(src_path), sr=None, mono=False)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.T
    wav_path = wav_dir / f"{src_path.stem}.wav"
    sf.write(str(wav_path), audio, sample_rate, format="WAV", subtype="PCM_16")
    return wav_path, sample_rate


def write_manifest(base_dir: Path, rows: list[dict]) -> Path:
    manifest_path = base_dir / "manifest.csv"
    fieldnames = [
        "voice_name",
        "gender",
        "score",
        "native_score",
        "url",
        "base_name",
        "original_path",
        "wav_path",
    ]
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download public audio URLs, convert them to WAV, and write a manifest."
    )
    parser.add_argument(
        "--output-dir",
        default=r"d:\work\tts\SpeechJudge\infer\examples\user_demo\fr_group_batch",
        help="Base output directory containing original/ and wav/ subfolders.",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    original_dir = base_dir / "original"
    wav_dir = base_dir / "wav"
    original_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total = len(ITEMS)
    for idx, item in enumerate(ITEMS, start=1):
        base_name = (
            f"{idx:02d}_{slugify(item['voice_name'])}"
            f"_hexin{item['hexin_score']}_native{item['native_score']}"
        )
        original_path = original_dir / f"{base_name}.mp3"
        print(f"[{idx}/{total}] Downloading {item['voice_name']} -> {original_path.name}")
        urlretrieve(item["url"], str(original_path))
        wav_path, sample_rate = convert_to_wav(original_path, wav_dir)
        print(f"  WAV: {wav_path.name} ({sample_rate} Hz)")
        manifest_rows.append(
            {
                "voice_name": item["voice_name"],
                "gender": item["gender"],
                "score": item["hexin_score"],
                "native_score": item["native_score"],
                "url": item["url"],
                "base_name": base_name,
                "original_path": str(original_path),
                "wav_path": str(wav_path),
            }
        )

    manifest_path = write_manifest(base_dir, manifest_rows)
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
