"""Persistent local API for SpeechJudge single-audio scoring.

Start once, keep the model on GPU, and score many requests through HTTP.

Examples:
    uvicorn api_service:app --host 0.0.0.0 --port 8000

    curl -X POST "http://127.0.0.1:8000/score-path" ^
      -H "Content-Type: application/json" ^
      -d "{\"audio_path\":\"D:\\\\audio.wav\",\"target_text\":\"Hello world.\"}"
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional
from urllib import request as urlrequest
from urllib.parse import urlparse

import librosa
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

_INFER_DIR = Path(__file__).resolve().parent
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))

from main_grm import auto_max_new_tokens_for_device, load_model
from score_single_wav import score_wav, score_wav_compact, score_wav_fast


class ScorePathRequest(BaseModel):
    audio_path: str = Field(..., description="Absolute or relative path to local audio.")
    target_text: str = Field(..., min_length=1, description="Expected transcript.")
    max_new_tokens: Optional[int] = Field(
        default=None, description="Override decode cap. Default is auto from GPU VRAM."
    )
    analysis: bool = Field(
        default=False,
        description="If true, ask for analysis text before the final score. Slower.",
    )
    mode: Optional[Literal["fast", "compact", "analysis"]] = Field(
        default=None,
        description="Scoring mode. Default is compact for faster structured scores.",
    )


class ScoreUrlRequest(BaseModel):
    audio_url: str = Field(..., description="Public HTTP(S) URL to an audio file.")
    target_text: str = Field(..., min_length=1, description="Expected transcript.")
    max_new_tokens: Optional[int] = Field(
        default=None, description="Override decode cap. Default is auto from GPU VRAM."
    )
    analysis: bool = Field(
        default=False,
        description="If true, ask for analysis text before the final score. Slower.",
    )
    mode: Optional[Literal["fast", "compact", "analysis"]] = Field(
        default=None,
        description="Scoring mode. Default is compact for faster structured scores.",
    )


def download_audio_to_temp(audio_url: str) -> Path:
    parsed = urlparse(audio_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("audio_url must start with http:// or https://")

    suffix = Path(parsed.path).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        with urlrequest.urlopen(audio_url, timeout=120) as resp:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
    return tmp_path


def convert_audio_to_temp_wav(src_path: Path) -> Path:
    audio, sample_rate = librosa.load(str(src_path), sr=None, mono=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = Path(tmp.name)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.T
    sf.write(str(wav_path), audio, sample_rate, format="WAV", subtype="PCM_16")
    return wav_path


class ModelServer:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.model_path = None
        self.cuda_device = None
        self.lock = threading.Lock()
        self.loaded_at = None

    def load_once(
        self,
        model_path: str,
        cuda_device: Optional[int] = None,
        thinker: bool = False,
    ) -> None:
        if self.model is not None:
            return
        model, processor = load_model(
            model_path,
            is_omni=not thinker,
            cuda_device=cuda_device,
        )
        model.eval()
        self.model = model
        self.processor = processor
        self.model_path = model_path
        self.cuda_device = model.device.index
        self.loaded_at = time.time()

    def auto_tokens(self) -> Optional[int]:
        if self.cuda_device is None:
            return None
        return auto_max_new_tokens_for_device(self.cuda_device)

    def effective_response_max_new_tokens(
        self, max_new_tokens: Optional[int], mode: str
    ) -> Optional[int]:
        base = max_new_tokens or self.auto_tokens()
        if base is None:
            return None
        if mode == "fast":
            return min(base, 16)
        if mode == "compact":
            return min(base, 64)
        return base

    def score_path(
        self,
        audio_path: str,
        target_text: str,
        max_new_tokens: Optional[int] = None,
        thinker: bool = False,
        analysis: bool = False,
        mode: Optional[str] = None,
    ) -> dict:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded.")

        audio_file = Path(audio_path)
        if not audio_file.is_file():
            raise FileNotFoundError(audio_path)

        mode = mode or ("analysis" if analysis else "compact")

        with self.lock, torch.inference_mode():
            if mode == "analysis":
                score, details = score_wav(
                    self.processor,
                    self.model,
                    target_text,
                    str(audio_file),
                    is_omni=not thinker,
                    max_new_tokens=max_new_tokens,
                )
                sub_scores = None
                raw_response = details
            elif mode == "fast":
                score, raw_response = score_wav_fast(
                    self.processor,
                    self.model,
                    target_text,
                    str(audio_file),
                    is_omni=not thinker,
                    max_new_tokens=max_new_tokens,
                )
                details = None
                sub_scores = None
            else:
                score, sub_scores, raw_response = score_wav_compact(
                    self.processor,
                    self.model,
                    target_text,
                    str(audio_file),
                    is_omni=not thinker,
                    max_new_tokens=max_new_tokens,
                )
                details = sub_scores

        return {
            "score": score,
            "details": details,
            "sub_scores": sub_scores,
            "raw_response": raw_response,
            "audio_path": str(audio_file),
            "target_text": target_text,
            "max_new_tokens": self.effective_response_max_new_tokens(
                max_new_tokens, mode
            ),
            "audio_format": audio_file.suffix.lower(),
            "wav_recommended": audio_file.suffix.lower() != ".wav",
            "analysis": mode == "analysis",
            "mode": mode,
        }


SERVER = ModelServer()


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_path = os.environ.get("SPEECHJUDGE_MODEL_PATH", "pretrained/SpeechJudge-GRM")
    cuda_raw = os.environ.get("SPEECHJUDGE_CUDA_DEVICE")
    cuda_device = int(cuda_raw) if cuda_raw is not None else None
    thinker = os.environ.get("SPEECHJUDGE_THINKER", "").lower() in {"1", "true", "yes"}
    SERVER.load_once(model_path=model_path, cuda_device=cuda_device, thinker=thinker)
    yield


app = FastAPI(
    title="SpeechJudge API",
    version="0.1.0",
    description="Persistent local API for single-audio naturalness scoring.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    if SERVER.model is None:
        return {"status": "loading"}
    return {
        "status": "ready",
        "model_path": SERVER.model_path,
        "cuda_device": SERVER.cuda_device,
        "auto_max_new_tokens": SERVER.auto_tokens(),
        "loaded_at": SERVER.loaded_at,
        "default_mode": "compact",
        "recommended_request_max_new_tokens": {"fast": 16, "compact": 64, "analysis": 256},
        "model_dtype": str(SERVER.model.dtype) if SERVER.model is not None else None,
    }


@app.post("/score-path")
def score_path(payload: ScorePathRequest) -> dict:
    try:
        return SERVER.score_path(
            audio_path=payload.audio_path,
            target_text=payload.target_text,
            max_new_tokens=payload.max_new_tokens,
            analysis=payload.analysis,
            mode=payload.mode,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Audio file not found: {exc}") from exc
    except Exception as exc:  # pragma: no cover - runtime/model exceptions
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/score-url")
def score_url(payload: ScoreUrlRequest) -> dict:
    src_path = None
    wav_path = None
    try:
        src_path = download_audio_to_temp(payload.audio_url)
        wav_path = convert_audio_to_temp_wav(src_path)
        result = SERVER.score_path(
            audio_path=str(wav_path),
            target_text=payload.target_text,
            max_new_tokens=payload.max_new_tokens,
            analysis=payload.analysis,
            mode=payload.mode,
        )
        result["audio_url"] = payload.audio_url
        result["downloaded_audio_format"] = src_path.suffix.lower()
        result["server_transcoded_to_wav"] = True
        return result
    except Exception as exc:  # pragma: no cover - runtime/network/model exceptions
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if src_path is not None and src_path.exists():
            src_path.unlink(missing_ok=True)
        if wav_path is not None and wav_path.exists():
            wav_path.unlink(missing_ok=True)


@app.post("/score-upload")
async def score_upload(
    target_text: str = Form(...),
    max_new_tokens: Optional[int] = Form(default=None),
    analysis: bool = Form(default=False),
    mode: Optional[str] = Form(default=None),
    audio: UploadFile = File(...),
) -> dict:
    suffix = Path(audio.filename or "upload.bin").suffix or ".bin"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        return SERVER.score_path(
            audio_path=str(tmp_path),
            target_text=target_text,
            max_new_tokens=max_new_tokens,
            analysis=analysis,
            mode=mode,
        )
    except Exception as exc:  # pragma: no cover - runtime/model exceptions
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await audio.close()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
