from __future__ import annotations

from typing import Tuple

import torch

from main_grm import effective_max_new_tokens
from utils import build_cot_conversation, build_qwen_omni_inputs, extract_rating


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
    Pairwise compare like ``infer/main_grm.compare_wavs``, but with ``do_sample=False``
    for more stable repeated comparisons during sorting.
    """
    conversion = build_cot_conversation(target_text, wav_path_a, wav_path_b)
    omni_inputs = build_qwen_omni_inputs(processor, conversion)
    omni_inputs = omni_inputs.to(model.device).to(model.dtype)
    prompt_length = omni_inputs["input_ids"].shape[1]

    gen_max = effective_max_new_tokens(max_new_tokens, model)

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
    text = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return extract_rating(text[0])


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
    conversations = [build_cot_conversation(target_text, a, b) for a, b in pairs]
    omni_inputs = build_qwen_omni_inputs(processor, conversations)
    prepared = _prepare_omni_inputs_for_model(omni_inputs, model)
    prompt_length = int(prepared["input_ids"].shape[1])

    gen_max = effective_max_new_tokens(max_new_tokens, model)

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
    return [_rating_to_pref(extract_rating(t)[0]) for t in decoded]
