"""
vLLM compare pipeline aligned with infer/main_grm_vllm.py (official example).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from transformers import Qwen2_5OmniProcessor

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from infer.utils import (  # noqa: E402
    build_cot_conversation,
    download_speechjudge_grm,
    extract_rating,
    load_grm_model_path,
)

ProgressFn = Optional[Callable[[str, dict[str, Any]], None]]


def load_model(model_path: str):
    """Same as infer/main_grm_vllm.load_model."""
    from vllm import LLM, SamplingParams

    print("Downloading model to {}...".format(model_path))
    download_speechjudge_grm(model_path)

    print("Loading model...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        max_model_len=5632,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": 2},
        seed=0,
        gpu_memory_utilization=0.5,
    )
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=50, max_tokens=1024
    )
    return processor, llm, sampling_params


def compare_wavs(
    processor: Qwen2_5OmniProcessor,
    model: Any,
    vllm_sampling_params: Any,
    target_text: str,
    wav_path_a: str,
    wav_path_b: str,
    num_of_generation: int = 10,
    on_progress: ProgressFn = None,
) -> list[tuple[Optional[dict[str, str]], str]]:
    """
    Mirrors infer/main_grm_vllm.compare_wavs; optional on_progress(step, extra).
    """
    import librosa

    conversion = build_cot_conversation(target_text, wav_path_a, wav_path_b)

    text = processor.apply_chat_template(
        conversion, add_generation_prompt=True, tokenize=False
    )
    assert len(text) == 1
    text = text[0]

    if on_progress:
        on_progress("loading_audio", {"message": "librosa.load both channels"})

    audio_data = {
        "audio": [
            librosa.load(wav_path_a, sr=None),
            librosa.load(wav_path_b, sr=None),
        ]
    }
    vllm_query = {"prompt": text, "multi_modal_data": audio_data}

    queries = [vllm_query for _ in range(num_of_generation)]
    if on_progress:
        on_progress(
            "generating",
            {"current": 0, "total": num_of_generation, "message": "vLLM generate"},
        )

    vllm_outputs = model.generate(queries, vllm_sampling_params)
    assert len(vllm_outputs) == num_of_generation

    result_list: list[tuple[Optional[dict[str, str]], str]] = []
    for i, o in enumerate(vllm_outputs):
        out_text = o.outputs[0].text
        rating, result = extract_rating(out_text)
        result_list.append((rating, result))
        if on_progress:
            on_progress(
                "generating",
                {
                    "current": i + 1,
                    "total": num_of_generation,
                    "message": f"decoded {i + 1}/{num_of_generation}",
                },
            )

    if num_of_generation == 1:
        return [result_list[0]]

    return result_list


def aggregate_like_official_main(
    result_list: list[tuple[Optional[dict[str, str]], str]],
) -> dict[str, Any]:
    """Same aggregation as infer/main_grm_vllm.py __main__ block."""
    audio_a_scores: list[float] = []
    audio_b_scores: list[float] = []
    cot_details: list[str] = []
    errors: list[str] = []

    for i, (rating, result) in enumerate(result_list):
        if rating is None:
            errors.append(f"generation {i + 1}: no rating parsed")
            continue
        a, b = rating["output_a"], rating["output_b"]
        audio_a_scores.append(float(a))
        audio_b_scores.append(float(b))
        cot_details.append(result)

    if not audio_a_scores:
        return {
            "final_result": "error",
            "score_a_avg": None,
            "score_b_avg": None,
            "generations": len(result_list),
            "cot_details": cot_details,
            "parse_errors": errors,
        }

    score_a = sum(audio_a_scores) / len(audio_a_scores)
    score_b = sum(audio_b_scores) / len(audio_b_scores)
    final = "A" if score_a > score_b else "B" if score_a < score_b else "Tie"
    return {
        "final_result": final,
        "score_a_avg": score_a,
        "score_b_avg": score_b,
        "generations_used": len(audio_a_scores),
        "generations_total": len(result_list),
        "cot_details": cot_details,
        "parse_errors": errors,
    }


def resolve_model_path() -> str:
    return load_grm_model_path()
