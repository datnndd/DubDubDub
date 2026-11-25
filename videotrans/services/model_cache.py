# -*- coding: utf-8 -*-
"""In-memory model cache for TTS preview and voice cloning inference engines.

Eliminates repetitive disk-to-GPU/CPU reloading of large model checkpoints
during interactive voice auditioning and voice cloning tests.
"""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
import threading
from typing import Any, Optional, Tuple

logger = logging.getLogger("videotrans.model_cache")

_lock = threading.Lock()
_vieneu_cache: dict[Tuple[str, str], Any] = {}
_omnivoice_cache: dict[Tuple[str, str], Any] = {}


def _configure_offline_hf() -> None:
    """Ensure HuggingFace Hub does not perform redundant network checks when local models exist."""
    try:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    except Exception:
        pass


def get_cached_vieneu_engine(
    device: str = "auto",
    backend: str = "auto",
    max_batch_size: int = 4,
) -> Any:
    """Return a cached Vieneu instance for the requested device and backend."""
    _configure_offline_hf()
    from videotrans.tts._vieneu_compat import setup_vieneu_environment
    setup_vieneu_environment()

    dev = str(device or "auto").lower()
    bak = str(backend or "auto").lower()
    if dev in ("auto", ""):
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    if bak in ("auto", ""):
        bak = "pytorch" if "cuda" in dev else "onnx"

    cache_key = (dev, bak)
    with _lock:
        if cache_key in _vieneu_cache:
            engine = _vieneu_cache[cache_key]
            cur_max = getattr(engine, "max_batch_size", None)
            if isinstance(cur_max, int) and not isinstance(cur_max, bool) and cur_max < max_batch_size:
                engine.max_batch_size = max_batch_size
            return engine

        logger.info("Initializing and caching VieNeu engine (device=%s, backend=%s, max_batch=%s)...", dev, bak, max_batch_size)
        from vieneu import Vieneu
        engine = Vieneu(
            mode="v3turbo",
            device=dev,
            backend=bak,
            max_batch_size=max_batch_size,
        )
        _vieneu_cache[cache_key] = engine
        return engine


def evict_vieneu_engine(device: Optional[str] = None) -> None:
    """Evict and close cached VieNeu engine(s)."""
    with _lock:
        to_del = []
        for k, engine in _vieneu_cache.items():
            if device is None or device in k[0]:
                try:
                    if hasattr(engine, "close"):
                        engine.close()
                except Exception as exc:
                    logger.debug("Error closing VieNeu engine: %s", exc)
                to_del.append(k)
        for k in to_del:
            _vieneu_cache.pop(k, None)


def load_omnivoice_model(model_dir: Path | str, device: str, dtype: Any) -> Any:
    from huggingface_hub import try_to_load_from_cache
    from omnivoice import OmniVoice

    asr_model = "openai/whisper-base"
    cached_weights = try_to_load_from_cache(asr_model, "model.safetensors")
    if isinstance(cached_weights, str):
        snapshot = Path(cached_weights).parent
        if all((snapshot / name).is_file() for name in (
            "config.json", "preprocessor_config.json", "tokenizer.json", "generation_config.json",
        )):
            asr_model = str(snapshot)
    # Whisper loads only for automatic transcription. Keep VRAM available for OmniVoice.
    return OmniVoice.from_pretrained(
        str(model_dir), device_map=device, dtype=dtype,
        asr_model_name=asr_model, asr_device="cpu",
    )


def get_cached_omnivoice_model(
    model_dir: Path | str,
    device: str = "auto",
    dtype: Any = None,
) -> Any:
    """Return a cached OmniVoice model instance."""
    _configure_offline_hf()
    model_path = Path(model_dir).resolve()
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"OmniVoice model not installed at {model_path}")

    dev = str(device or "auto").lower()
    cache_key = (str(model_path), dev, str(dtype))
    with _lock:
        if cache_key in _omnivoice_cache:
            return _omnivoice_cache[cache_key]

        logger.info("Loading and caching OmniVoice model from %s on %s...", model_path, dev)
        model = load_omnivoice_model(model_path, device, dtype)
        _omnivoice_cache[cache_key] = model
        return model


def evict_omnivoice_model(model_dir: Optional[Path | str] = None, device: Optional[str] = None) -> None:
    """Evict cached OmniVoice model(s)."""
    with _lock:
        to_del = []
        m_str = str(Path(model_dir).resolve()) if model_dir else None
        for k in list(_omnivoice_cache.keys()):
            if (m_str is None or k[0] == m_str) and (device is None or device in k[1]):
                to_del.append(k)
        for k in to_del:
            _omnivoice_cache.pop(k, None)


def clear_model_cache() -> None:
    """Evict all cached models and reclaim system and GPU memory."""
    evict_vieneu_engine()
    evict_omnivoice_model()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
