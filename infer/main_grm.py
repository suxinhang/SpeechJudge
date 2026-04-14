import importlib.util
import os

import torch
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


from utils import (
    build_cot_conversation,
    build_qwen_omni_inputs,
    download_hugginface_model,
    count_parameters,
    extract_rating,
)


def resolve_cuda_device_index(cuda_device=None) -> int:
    """Logical CUDA index (respects ``CUDA_VISIBLE_DEVICES``)."""
    if cuda_device is None:
        cuda_device = int(os.environ.get("SPEECHJUDGE_CUDA_DEVICE", "0"))
    return int(cuda_device)


def auto_torch_dtype_for_device(cuda_device_index: int) -> torch.dtype:
    """
    Pick a runtime-friendly dtype for the current GPU generation.

    TITAN Xp (Pascal, compute capability 6.1) does not have native BF16 support,
    so ``float16`` is a better fit than ``bfloat16`` there.
    """
    major, _minor = torch.cuda.get_device_capability(cuda_device_index)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def auto_max_new_tokens_for_device(cuda_device_index: int) -> int:
    """
    Pick a conservative ``max_new_tokens`` from total VRAM (decode KV cache).

    Buckets are heuristics for ~10B-class multimodal models; override with CLI if needed.
    """
    props = torch.cuda.get_device_properties(cuda_device_index)
    total_gb = props.total_memory / (1024.0**3)
    if total_gb < 13:
        return 256
    if total_gb < 20:
        return 384
    if total_gb < 34:
        return 512
    if total_gb < 48:
        return 768
    return 1024


def effective_max_new_tokens(max_new_tokens, model) -> int:
    """Use explicit cap if set, else choose from the GPU that hosts ``model``."""
    if max_new_tokens is not None:
        return min(int(max_new_tokens), 1024)
    if not torch.cuda.is_available():
        return 512
    idx = model.device.index
    if idx is None:
        idx = torch.cuda.current_device()
    return min(auto_max_new_tokens_for_device(idx), 1024)


def load_model(model_path, is_omni=True, cuda_device=None):
    """
    Load weights on GPU when CUDA is available.

    ``cuda_device``: GPU index (e.g. 0). If None, uses env ``SPEECHJUDGE_CUDA_DEVICE`` or 0.
    All parameters are placed on that CUDA device via ``device_map``.
    """
    if is_omni:
        qwen_cls = Qwen2_5OmniForConditionalGeneration
    else:
        qwen_cls = Qwen2_5OmniThinkerForConditionalGeneration

    download_hugginface_model("RMSnow/SpeechJudge-GRM", model_path)

    attn_impl = (
        "flash_attention_2"
        if importlib.util.find_spec("flash_attn") is not None
        else "sdpa"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available: PyTorch does not see a GPU. "
            "Install a CUDA build of PyTorch (see https://pytorch.org/get-started/locally/)."
        )

    cuda_device = resolve_cuda_device_index(cuda_device)
    if cuda_device < 0 or cuda_device >= torch.cuda.device_count():
        raise ValueError(
            f"cuda_device={cuda_device} invalid; found {torch.cuda.device_count()} GPU(s)."
        )

    device_map = {"": cuda_device}
    torch_dtype = auto_torch_dtype_for_device(cuda_device)
    print(
        f"Loading model on GPU {cuda_device}: {torch.cuda.get_device_name(cuda_device)} "
        f"(attention: {attn_impl}, dtype: {torch_dtype})..."
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model = qwen_cls.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
    )

    # print(model)
    print(f"#Params of Model: {count_parameters(model)}")
    return model, processor


def compare_wavs(
    processor,
    model,
    target_text,
    wav_path_a,
    wav_path_b,
    is_omni=True,
    max_new_tokens=None,
):
    conversion = build_cot_conversation(target_text, wav_path_a, wav_path_b)
    omni_inputs = build_qwen_omni_inputs(processor, conversion)

    omni_inputs = omni_inputs.to(model.device).to(model.dtype)
    prompt_length = omni_inputs["input_ids"].shape[1]

    gen_max = effective_max_new_tokens(max_new_tokens, model)

    if is_omni:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=True,
            return_audio=False,
            max_new_tokens=gen_max,
        )  # [1, T]
    else:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=True,
            max_new_tokens=gen_max,
            eos_token_id=[151645],
            pad_token_id=151643,
        )  # [1, T]
    text_ids = text_ids[:, prompt_length:]

    text = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    rating, result = extract_rating(text[0])
    return rating, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SpeechJudge-GRM pairwise demo (GPU).")
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
    cli = parser.parse_args()

    model_path = "pretrained/SpeechJudge-GRM"
    model, processor = load_model(model_path, cuda_device=cli.cuda_device)

    if cli.max_new_tokens is None:
        idx = model.device.index
        if idx is None:
            idx = torch.cuda.current_device()
        gb = torch.cuda.get_device_properties(idx).total_memory / (1024.0**3)
        tok = auto_max_new_tokens_for_device(idx)
        print(
            f"[infer] GPU {idx} ~{gb:.1f} GiB total -> max_new_tokens={tok} "
            f"(override: --max-new-tokens)"
        )

    target_text = "The worn leather, once supple and inviting, now hangs limp and lifeless. Its time has passed, like autumn leaves surrendering to winter's chill. I shall cast it aside, making way for new beginnings and fresh possibilities."
    wav_path_a = "examples/wav_a.wav"
    wav_path_b = "examples/wav_b.wav"

    rating, result = compare_wavs(
        processor,
        model,
        target_text,
        wav_path_a,
        wav_path_b,
        max_new_tokens=cli.max_new_tokens,
    )

    if rating is None:
        print("\n[Final Result] (could not parse scores)")
    else:
        score_A = float(rating["output_a"])
        score_B = float(rating["output_b"])
        final_result = (
            "A" if score_A > score_B else "B" if score_A < score_B else "Tie"
        )
        print(f"\n[Final Result] {final_result}")
        print(f"Score of Audio A: {score_A}, Score of Audio B: {score_B}")
    print("\n", "-" * 15, f"Details", "-" * 15, "\n")
    print(result)
