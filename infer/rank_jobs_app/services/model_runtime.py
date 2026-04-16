from __future__ import annotations

import threading
from typing import Optional, Tuple

import torch

from main_grm import load_model


class SharedModel:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._loaded = False

    def load_once(
        self,
        *,
        model_path: str,
        cuda_device: Optional[int],
        thinker: bool,
    ) -> None:
        with self._lock:
            if self._loaded:
                return
            model, processor = load_model(
                model_path,
                is_omni=not thinker,
                cuda_device=cuda_device,
            )
            model.eval()
            self._model = model
            self._processor = processor
            self._loaded = True

    def get(self) -> Tuple[object, object]:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Model is not loaded.")
        return self._model, self._processor

    def infer_lock(self):
        return self._lock

    @torch.inference_mode()
    def context(self):
        return torch.inference_mode()


MODEL = SharedModel()
