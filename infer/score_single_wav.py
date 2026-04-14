"""
Single-file naturalness scoring (1–10) with SpeechJudge-GRM.

Standalone entry point: does not modify ``main_grm.py`` / ``utils.py``.
Reuses ``load_model`` and ``build_qwen_omni_inputs`` only.

Example (from ``infer/``):

    python score_single_wav.py examples/wav_a.wav --target "Hello world."

The GRM is trained mainly on pairwise preferences; absolute scores are indicative
unless calibrated on your own data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

_INFER_DIR = Path(__file__).resolve().parent
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))

from main_grm import (
    auto_max_new_tokens_for_device,
    effective_max_new_tokens,
    load_model,
)
from utils import build_qwen_omni_inputs


def _prepare_omni_inputs(processor, model, conversations):
    omni_inputs = build_qwen_omni_inputs(processor, conversations)
    prepared = {}
    for key, value in omni_inputs.items():
        if hasattr(value, "to"):
            value = value.to(model.device)
            if torch.is_tensor(value) and value.is_floating_point():
                value = value.to(model.dtype)
        prepared[key] = value
    return prepared


def _resolve_mode_max_new_tokens(max_new_tokens, model, mode: str) -> int:
    gen_max = effective_max_new_tokens(max_new_tokens, model)
    if mode == "fast":
        return min(gen_max, 16)
    if mode == "compact":
        return min(gen_max, 64)
    return gen_max


def build_single_score_conversation(target_text: str, wav_path: str):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "We are evaluating the naturalness of one Text-to-Speech output. The audio should speak the target text naturally.",
                },
                {"type": "text", "text": f"Target text: {target_text}"},
                {"type": "text", "text": "Output:"},
                {"type": "audio", "audio": wav_path},
                {
                    "type": "text",
                    "text": (
                        "Evaluate the audio using these criteria: Prosody and Intonation, "
                        "Pacing and Rhythm, Articulation and Clarity, and Overall Naturalness. "
                        "Write a brief analysis, then end your reply with exactly one line in this "
                        "format (X must be an integer from 1 to 10): Overall score: X"
                    ),
                },
            ],
        },
    ]


def build_single_score_compact_conversation(target_text: str, wav_path: str):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "We are evaluating one Text-to-Speech output against the target text.",
                },
                {"type": "text", "text": f"Target text: {target_text}"},
                {"type": "text", "text": "Output:"},
                {"type": "audio", "audio": wav_path},
                {
                    "type": "text",
                    "text": (
                        "Return only one compact JSON object with numeric scores from 1.0 to 10.0. "
                        'Use exactly these keys: {"naturalness": x, "accuracy": x, "emotion": x, "overall": x}. '
                        "Use one decimal place and no extra text."
                    ),
                },
            ],
        },
    ]


def build_single_score_only_conversation(target_text: str, wav_path: str):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "We are evaluating the naturalness of one Text-to-Speech output. The audio should speak the target text naturally.",
                },
                {"type": "text", "text": f"Target text: {target_text}"},
                {"type": "text", "text": "Output:"},
                {"type": "audio", "audio": wav_path},
                {
                    "type": "text",
                    "text": (
                        "Score the audio naturalness from 1 to 10. "
                        "Reply with only one number such as 7 or 8.5, and nothing else."
                    ),
                },
            ],
        },
    ]


def extract_single_score(result: str):
    text = result.replace("**", "")
    m = re.search(r"Overall score:\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not m:
        return None, result
    value = float(m.group(1))
    if value < 1.0 or value > 10.0:
        return None, result
    return value, result


def extract_number_only_score(result: str):
    text = result.replace("**", "").strip()
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", text)
    if not m:
        return None, result
    value = float(m.group(1))
    if value < 1.0 or value > 10.0:
        return None, result
    return value, result


def extract_compact_scores(result: str):
    text = result.replace("**", "").strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None, None, result
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None, result

    normalized = {}
    for key in ("naturalness", "accuracy", "emotion", "overall"):
        if key not in payload:
            return None, None, result
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            return None, None, result
        if value < 1.0 or value > 10.0:
            return None, None, result
        normalized[key] = round(value, 1)

    return normalized["overall"], normalized, result


def score_wav(
    processor,
    model,
    target_text: str,
    wav_path: str,
    is_omni: bool = True,
    max_new_tokens=None,
):
    conversations = build_single_score_conversation(target_text, wav_path)
    omni_inputs = _prepare_omni_inputs(processor, model, conversations)
    prompt_length = omni_inputs["input_ids"].shape[1]

    gen_max = _resolve_mode_max_new_tokens(max_new_tokens, model, mode="analysis")

    if is_omni:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=True,
            return_audio=False,
            max_new_tokens=gen_max,
        )
    else:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=True,
            max_new_tokens=gen_max,
            eos_token_id=[151645],
            pad_token_id=151643,
        )
    text_ids = text_ids[:, prompt_length:]
    decoded = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return extract_single_score(decoded[0])


def score_wav_fast(
    processor,
    model,
    target_text: str,
    wav_path: str,
    is_omni: bool = True,
    max_new_tokens=None,
):
    conversations = build_single_score_only_conversation(target_text, wav_path)
    omni_inputs = _prepare_omni_inputs(processor, model, conversations)
    prompt_length = omni_inputs["input_ids"].shape[1]

    gen_max = _resolve_mode_max_new_tokens(max_new_tokens, model, mode="fast")

    if is_omni:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=False,
            return_audio=False,
            max_new_tokens=gen_max,
        )
    else:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=False,
            max_new_tokens=gen_max,
            eos_token_id=[151645],
            pad_token_id=151643,
        )
    text_ids = text_ids[:, prompt_length:]
    decoded = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return extract_number_only_score(decoded[0])


def score_wav_compact(
    processor,
    model,
    target_text: str,
    wav_path: str,
    is_omni: bool = True,
    max_new_tokens=None,
):
    conversations = build_single_score_compact_conversation(target_text, wav_path)
    omni_inputs = _prepare_omni_inputs(processor, model, conversations)
    prompt_length = omni_inputs["input_ids"].shape[1]

    gen_max = _resolve_mode_max_new_tokens(max_new_tokens, model, mode="compact")

    if is_omni:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=False,
            return_audio=False,
            max_new_tokens=gen_max,
        )
    else:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=False,
            max_new_tokens=gen_max,
            eos_token_id=[151645],
            pad_token_id=151643,
        )
    text_ids = text_ids[:, prompt_length:]
    decoded = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return extract_compact_scores(decoded[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score one WAV for TTS naturalness (1–10) with SpeechJudge-GRM."
    )
    parser.add_argument("wav", type=str, help="Path to audio file (e.g. WAV).")
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="Target transcript the TTS was supposed to speak.",
    )
    parser.add_argument(
        "--model-path",
        default="pretrained/SpeechJudge-GRM",
        help="Local directory for the checkpoint (default: pretrained/SpeechJudge-GRM).",
    )
    parser.add_argument(
        "--thinker",
        action="store_true",
        help="Load Qwen2_5OmniThinkerForConditionalGeneration instead of full Omni.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        metavar="N",
        help="CUDA device index (default: env SPEECHJUDGE_CUDA_DEVICE or 0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        metavar="T",
        help="Cap decode length (max 1024). Default: auto from GPU VRAM.",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.is_file():
        print(f"error: file not found: {wav_path}", file=sys.stderr)
        return 1

    model, processor = load_model(
        args.model_path,
        is_omni=not args.thinker,
        cuda_device=args.cuda_device,
    )

    if args.max_new_tokens is None:
        idx = model.device.index
        if idx is None:
            idx = torch.cuda.current_device()
        gb = torch.cuda.get_device_properties(idx).total_memory / (1024.0**3)
        tok = auto_max_new_tokens_for_device(idx)
        print(
            f"[infer] GPU {idx} ~{gb:.1f} GiB total -> max_new_tokens={tok} "
            f"(override: --max-new-tokens)"
        )

    score, details = score_wav(
        processor,
        model,
        args.target,
        str(wav_path),
        is_omni=not args.thinker,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n[Overall score] {score if score is not None else '(parse failed)'}")
    print("\n" + "-" * 15 + " Details " + "-" * 15 + "\n")
    print(details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
