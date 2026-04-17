from __future__ import annotations

from typing import Tuple

import torch

from main_grm import effective_max_new_tokens
from utils import build_cot_conversation, build_qwen_omni_inputs, extract_rating


def _build_ab_only_conversation(target_text: str, wav_path_a: str, wav_path_b: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You compare two TTS audios for naturalness. "
                        "Reply with exactly one uppercase letter: A or B."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Choose the more natural audio for the target text.",
                },
                {"type": "text", "text": f"Target text: {target_text}"},
                {"type": "text", "text": "Audio A:"},
                {"type": "audio", "audio": wav_path_a},
                {"type": "text", "text": "Audio B:"},
                {"type": "audio", "audio": wav_path_b},
                {
                    "type": "text",
                    "text": (
                        "Return only A if Audio A is better, or only B if Audio B is better. "
                        "Do not explain."
                    ),
                },
            ],
        },
    ]


def _resolve_pairwise_max_new_tokens(max_new_tokens: int | None, model: object) -> int:
    return min(effective_max_new_tokens(max_new_tokens, model), 4)


def _pref_from_text(text: str) -> int:
    stripped = text.strip().upper()
    if stripped.startswith("A"):
        return 1
    if stripped.startswith("B"):
        return -1
    rating, _raw = extract_rating(text)
    return _rating_to_pref(rating)


def _rating_to_pref(rating: dict | None) -> int:
    if rating is None:
        return 0
    try:
        a = float(rating["output_a"])
        b = float(rating["output_b"])
    except (KeyError, TypeError, ValueError):
        return 0
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def _prepare_omni_inputs_for_model(omni_inputs: dict, model: object) -> dict:
    prepared: dict = {}
    for key, value in omni_inputs.items():
        if hasattr(value, "to"):
            value = value.to(model.device)
            if torch.is_tensor(value) and value.is_floating_point():
                value = value.to(model.dtype)
        prepared[key] = value
    return prepared


def compare_wavs_deterministic(
    processor,
    model,
    target_text: str,
    wav_path_a: str,
    wav_path_b: str,
    *,
    is_omni: bool = True,
    max_new_tokens: int | None = None,
) -> Tuple[dict | None, str]:
    """
    Pairwise compare with a minimal A/B-only reply for faster, more stable sorting.
    """
    conversion = _build_ab_only_conversation(target_text, wav_path_a, wav_path_b)
    omni_inputs = build_qwen_omni_inputs(processor, conversion)
    prepared = _prepare_omni_inputs_for_model(omni_inputs, model)
    prompt_length = prepared["input_ids"].shape[1]

    gen_max = _resolve_pairwise_max_new_tokens(max_new_tokens, model)

    if is_omni:
        text_ids = model.generate(
            **prepared,
            use_audio_in_video=False,
            do_sample=False,
            return_audio=False,
            max_new_tokens=gen_max,
        )
    else:
        text_ids = model.generate(
            **prepared,
            use_audio_in_video=False,
            do_sample=False,
            max_new_tokens=gen_max,
            eos_token_id=[151645],
            pad_token_id=151643,
        )

    text_ids = text_ids[:, prompt_length:]
    text = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    pref = _pref_from_text(text[0])
    if pref > 0:
        return {"output_a": "1", "output_b": "0"}, text[0]
    if pref < 0:
        return {"output_a": "0", "output_b": "1"}, text[0]
    return None, text[0]


def pairwise_preference(
    processor,
    model,
    *,
    is_omni: bool,
    max_new_tokens: int | None,
    target_text: str,
    left_wav: str,
    right_wav: str,
) -> int:
    """
    Return +1 if left is better, -1 if right is better, 0 if tie/unparseable treated as tie.
    """
    rating, _raw = compare_wavs_deterministic(
        processor,
        model,
        target_text,
        left_wav,
        right_wav,
        is_omni=is_omni,
        max_new_tokens=max_new_tokens,
    )
    return _rating_to_pref(rating)


def pairwise_preferences_batched(
    processor,
    model,
    *,
    is_omni: bool,
    max_new_tokens: int | None,
    target_text: str,
    pairs: list[tuple[str, str]],
) -> list[int]:
    """Run one batched ``model.generate`` for many (left_wav, right_wav) pairs (real GPU parallelism)."""
    if not pairs:
        return []
    conversations = [_build_ab_only_conversation(target_text, a, b) for a, b in pairs]
    omni_inputs = build_qwen_omni_inputs(processor, conversations)
    prepared = _prepare_omni_inputs_for_model(omni_inputs, model)
    prompt_length = int(prepared["input_ids"].shape[1])

    gen_max = _resolve_pairwise_max_new_tokens(max_new_tokens, model)

    if is_omni:
        text_ids = model.generate(
            **prepared,
            use_audio_in_video=False,
            do_sample=False,
            return_audio=False,
            max_new_tokens=gen_max,
        )
    else:
        text_ids = model.generate(
            **prepared,
            use_audio_in_video=False,
            do_sample=False,
            return_audio=False,
            max_new_tokens=gen_max,
            eos_token_id=[151645],
            pad_token_id=151643,
        )

    gen_only = text_ids[:, prompt_length:]
    decoded = processor.batch_decode(
        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [_pref_from_text(text) for text in decoded]
