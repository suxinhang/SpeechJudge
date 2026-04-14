from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gdown
import librosa
import soundfile as sf


ITEMS = [
    {
        "voice_name": "Naos - Espanol",
        "gender": "M",
        "score": 10,
        "url": "https://drive.google.com/file/d/12dgFvKvLbJKwsRjSuortRO7tWTDdMftp/view?usp=drive_link",
        "base_name": "01_naos_es_m_score10",
    },
    {
        "voice_name": "Maia - Espanol",
        "gender": "F",
        "score": 9,
        "url": "https://drive.google.com/file/d/1i4ufraFKDEyyzBVWeA7dUKTXVDS7i2AE/view?usp=drive_link",
        "base_name": "02_maia_es_f_score9",
    },
    {
        "voice_name": "Jabbah - Espanol",
        "gender": "M",
        "score": 4,
        "url": "https://drive.google.com/file/d/1RUzzAp4KVUxMkfQcZsdB-2sQ15tM40hO/view?usp=drive_link",
        "base_name": "03_jabbah_es_m_score4",
    },
    {
        "voice_name": "Vega - Espanol",
        "gender": "M",
        "score": 10,
        "url": "https://drive.google.com/file/d/1QQ4msWyf_KjUbjczh9egZ3J9Bbtt-U9P/view?usp=drive_link",
        "base_name": "04_vega_es_m_score10",
    },
    {
        "voice_name": "Hatysa - Espanol",
        "gender": "F",
        "score": 5,
        "url": "https://drive.google.com/file/d/1UFmyN98zUx9UoirhF4pxidmg2Vjrj5Wy/view?usp=drive_link",
        "base_name": "05_hatysa_es_f_score5",
    },
    {
        "voice_name": "Libertas - Espanol",
        "gender": "F",
        "score": 3,
        "url": "https://drive.google.com/file/d/1IccXJvhYdy7BTvLoKFv8JKl6MWSHe3rH/view?usp=drive_link",
        "base_name": "06_libertas_es_f_score3",
    },
    {
        "voice_name": "Izar - Espanol",
        "gender": "F",
        "score": 1,
        "url": "https://drive.google.com/file/d/1W_-L5QfJhdSIORLb0YWZlXjCuR3xBN-w/view?usp=drive_link",
        "base_name": "07_izar_es_f_score1",
    },
    {
        "voice_name": "Fulu - Espanol",
        "gender": "F",
        "score": 6,
        "url": "https://drive.google.com/file/d/18lzoL9maEihNqrRcWBVTs1M_ESdCehLu/view?usp=drive_link",
        "base_name": "08_fulu_es_f_score6",
    },
    {
        "voice_name": "Deneb - Espanol",
        "gender": "M",
        "score": 2,
        "url": "https://drive.google.com/file/d/1B7F6oYtrZZNj0h3WfxiCP4NxuAXvsseS/view?usp=drive_link",
        "base_name": "09_deneb_es_m_score2",
    },
    {
        "voice_name": "Ginan - Espanol",
        "gender": "F",
        "score": 4,
        "url": "https://drive.google.com/file/d/1vyI0VheCq_S_0bKmTLgD6XIlGKge33O0/view?usp=drive_link",
        "base_name": "10_ginan_es_f_score4",
    },
    {
        "voice_name": "Altair - Espanol",
        "gender": "M",
        "score": 8,
        "url": "https://drive.google.com/file/d/1uTwpEGhvJucRY76v6SnTFLTnUc-bOJvX/view?usp=drive_link",
        "base_name": "11_altair_es_m_score8",
    },
    {
        "voice_name": "Betel - Espanol",
        "gender": "F",
        "score": 6,
        "url": "https://drive.google.com/file/d/10gmB7nBRwK-kTzmnBCCQF6a0hf6nwMWD/view?usp=drive_link",
        "base_name": "12_betel_es_f_score6",
    },
    {
        "voice_name": "Castor - Espanol",
        "gender": "M",
        "score": 7,
        "url": "https://drive.google.com/file/d/1F7Glg18t5kpHkFnoiMY8sZ0DZIlenzaK/view?usp=drive_link",
        "base_name": "13_castor_es_m_score7",
    },
]


def convert_to_wav(src_path: Path, wav_dir: Path) -> tuple[Path, int]:
    audio, sample_rate = librosa.load(str(src_path), sr=None, mono=False)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.T
    wav_path = wav_dir / f"{src_path.stem}.wav"
    sf.write(str(wav_path), audio, sample_rate, format="WAV", subtype="PCM_16")
    return wav_path, sample_rate


def write_manifest(base_dir: Path, rows: list[dict]) -> None:
    manifest_path = base_dir / "manifest.csv"
    fieldnames = [
        "voice_name",
        "gender",
        "score",
        "url",
        "base_name",
        "original_path",
        "wav_path",
    ]
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Google Drive audio files and convert them to WAV."
    )
    parser.add_argument(
        "--output-dir",
        default=r"d:\work\tts\SpeechJudge\infer\examples\user_demo\drive_batch",
        help="Base output directory containing original/ and wav/ subfolders.",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    original_dir = base_dir / "original"
    wav_dir = base_dir / "wav"
    original_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for idx, item in enumerate(ITEMS, start=1):
        url = item["url"]
        print(f"[{idx}/{len(ITEMS)}] Downloading: {url}")
        original_path = original_dir / f"{item['base_name']}.mp3"
        out = gdown.download(url=url, output=str(original_path), quiet=False)
        if not out:
            raise RuntimeError(f"Download failed: {url}")

        src_path = Path(out)
        wav_path, sample_rate = convert_to_wav(src_path, wav_dir)
        print(f"  saved original: {src_path}")
        print(f"  saved wav:      {wav_path} ({sample_rate} Hz)")
        manifest_rows.append(
            {
                "voice_name": item["voice_name"],
                "gender": item["gender"],
                "score": item["score"],
                "url": item["url"],
                "base_name": item["base_name"],
                "original_path": str(src_path),
                "wav_path": str(wav_path),
            }
        )

    write_manifest(base_dir, manifest_rows)
    print("\nDone.")
    print(f"Original files: {original_dir}")
    print(f"WAV files:      {wav_dir}")
    print(f"Manifest CSV:   {base_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
