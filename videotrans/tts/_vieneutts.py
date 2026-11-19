from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf

from videotrans.tts._base import BaseTTS
from videotrans.util.help_role import get_vieneu_custom_voice_path
from videotrans.services.voice_preview import resolve_vieneu_preset_voice, _gpu_thread_lock


@dataclass
class VieNeuTTS(BaseTTS):
    engine: Any = field(init=False, repr=False)
    _owns_engine: bool = field(default=True, init=False, repr=False)
    _voice_options_cache: Dict[Tuple[Any, ...], Tuple[Tuple[Any, ...], Dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        super().__post_init__()
        self.engine = self._create_engine()

    def _create_engine(self):
        from videotrans.tts._vieneu_compat import setup_vieneu_environment
        setup_vieneu_environment()

        device = "cuda" if self.is_cuda else "cpu"
        backend = "pytorch" if self.is_cuda else "onnx"

        vieneu_mod = sys.modules.get("vieneu")
        if vieneu_mod is None or isinstance(vieneu_mod, types.ModuleType):
            from videotrans.services.model_cache import get_cached_vieneu_engine
            self._owns_engine = False
            return get_cached_vieneu_engine(
                device=device,
                backend=backend,
                max_batch_size=4,
            )

        # Test double (e.g. SimpleNamespace) — instantiate and close directly
        self._owns_engine = True
        return vieneu_mod.Vieneu(
            mode="v3turbo",
            device=device,
            backend=backend,
            max_batch_size=4,
        )

    def _inference_options(
        self, item: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        role = str(item.get("role") or "").strip()
        item_style = str(item.get("style") or "").strip()
        if role.lower() == "clone":
            ref_wav, _ = self.get_ref_wav(item)
            cache_key: Tuple[Any, ...] = ("clone", ref_wav, item_style)
        else:
            cache_key = (role, item_style)

        cached = self._voice_options_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._resolve_inference_options(item, role, item_style)
        self._voice_options_cache[cache_key] = result
        return result

    def _resolve_inference_options(
        self, item: Dict[str, Any], role: str, item_style: str
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # Check if role maps to a recorded custom voice (cloned or designed)
        from videotrans.core import voice_store
        resolved = voice_store.resolve_voice_params(role)
        if resolved:
            tuning = dict(resolved.get("tuning_params") or {})
            is_design = resolved.get("kind") == "design"
            style = item_style or (str(tuning.get("style")) if is_design else "")
            temperature = float(tuning.get("temperature", 0.8))
            repetition_penalty = float(tuning.get("repetition_penalty", 1.2))
            top_p = float(tuning.get("top_p", 0.95))
            denoise = bool(tuning.get("denoise", True))

            eff_id = resolved.get("effective_voice_id") or resolved.get("id")
            cached_emb = voice_store.get_voice_embedding(eff_id) if eff_id else None

            if cached_emb and cached_emb[0] is not None:
                v_key: Tuple[Any, ...] = ("custom_cached", str(eff_id), style) if style else ("custom_cached", str(eff_id))
                opts: Dict[str, Any] = {"voice": {"speaker_emb": cached_emb[0], "codes": cached_emb[1]}}
                if style:
                    opts["style"] = style
                if is_design:
                    opts.update({"temperature": temperature, "repetition_penalty": repetition_penalty, "top_p": top_p})
                return v_key, opts

            if resolved.get("base_type") == "preset":
                preset_name = resolved.get("preset_voice") or resolved.get("external_voice_id") or role
                v_target = resolve_vieneu_preset_voice(preset_name, self.engine)
                v_key = ("preset", preset_name, style) if style else ("preset", preset_name)
                opts = {"voice": v_target}
                if style:
                    opts["style"] = style
                if is_design:
                    opts.update({"temperature": temperature, "repetition_penalty": repetition_penalty, "top_p": top_p})
                return v_key, opts

            ref_rel = resolved.get("ref_audio_path")
            if ref_rel:
                try:
                    ref_path = voice_store.get_voice_audio_path(ref_rel)
                    if ref_path.is_file():
                        v_key = ("custom_audio", ref_path.as_posix(), style) if style else ("custom_audio", ref_path.as_posix())
                        opts = {"ref_audio": ref_path.as_posix(), "denoise": denoise}
                        if style:
                            opts["style"] = style
                        if is_design:
                            opts.update({"temperature": temperature, "repetition_penalty": repetition_penalty, "top_p": top_p})
                        return v_key, opts
                except Exception:
                    pass

        if role.lower() == "clone":
            ref_wav, _ = self.get_ref_wav(item)
            if item_style:
                return ("clone", ref_wav, item_style), {"ref_audio": ref_wav, "style": item_style}
            return ("clone", ref_wav), {"ref_audio": ref_wav}

        custom_voice_path = get_vieneu_custom_voice_path(role)
        if custom_voice_path:
            reference_audio = Path(custom_voice_path)
            if not reference_audio.is_file():
                raise FileNotFoundError(f"VieNeu reference audio does not exist: {reference_audio}")
            if item_style:
                return ("custom", reference_audio.as_posix(), item_style), {"ref_audio": reference_audio.as_posix(), "style": item_style}
            return ("custom", reference_audio.as_posix()), {"ref_audio": reference_audio.as_posix()}

        v_target = resolve_vieneu_preset_voice(role, self.engine)
        if item_style:
            return ("preset", role, item_style), {"voice": v_target, "style": item_style}
        return ("preset", role), {"voice": v_target}

    def _exec(self) -> None:
        queue_by_voice: Dict[
            Tuple[Any, ...], List[Tuple[Dict[str, Any], Dict[str, Any]]]
        ] = defaultdict(list)
        for item in self.queue_tts:
            if not item.get("text", "").strip() or Path(item["filename"]).is_file():
                continue
            voice_key, options = self._inference_options(item)
            queue_by_voice[voice_key].append((item, options))

        chunk_size = max(4, int(getattr(self.engine, "max_batch_size", 4) or 4))

        try:
            completed = 0
            for items in queue_by_voice.values():
                if self._exit():
                    return
                queue = [item for item, _ in items]
                options = items[0][1]
                for start in range(0, len(queue), chunk_size):
                    if self._exit():
                        return
                    sub_queue = queue[start : start + chunk_size]
                    texts = [it["text"] for it in sub_queue]
                    try:
                        if not self._owns_engine:
                            with _gpu_thread_lock:
                                audios = self.engine.infer_batch(texts, **options)
                        else:
                            audios = self.engine.infer_batch(texts, **options)
                    except Exception:
                        if not self._owns_engine and self.is_cuda:
                            from videotrans.services.model_cache import evict_vieneu_engine
                            evict_vieneu_engine(device="cuda")
                        raise

                    if len(audios) != len(sub_queue):
                        raise RuntimeError(
                            "VieNeu-TTS returned an unexpected number of audio segments"
                        )
                    for item, audio in zip(sub_queue, audios):
                        if hasattr(audio, "detach"):
                            audio = audio.detach().cpu().numpy()
                        arr = np.asarray(audio, dtype=np.float32).squeeze()
                        if np.isnan(arr).any() or np.isinf(arr).any():
                            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
                        peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
                        if peak >= 1e-4:
                            arr = (arr / peak) * 0.90
                        sf.write(item["filename"], arr, self.engine.sample_rate)
                        completed += 1
                    self.signal(text=f"TTS[{completed}/{self.len}]")
            self.signal(text="TTS ended")
        finally:
            if self._owns_engine and hasattr(self.engine, "close"):
                self.engine.close()
