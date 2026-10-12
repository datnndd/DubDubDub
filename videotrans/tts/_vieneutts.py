from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import soundfile as sf

from videotrans.tts._base import BaseTTS
from videotrans.util.help_role import get_vieneu_custom_voice_path
from videotrans.services.voice_preview import resolve_vieneu_preset_voice


@dataclass
class VieNeuTTS(BaseTTS):
    engine: Any = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.engine = self._create_engine()

    def _create_engine(self):
        from videotrans.tts._vieneu_compat import setup_vieneu_environment
        setup_vieneu_environment()
        from vieneu import Vieneu

        device = "cuda" if self.is_cuda else "cpu"
        backend = "pytorch" if self.is_cuda else "onnx"
        return Vieneu(
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
                v_key = ("custom_cached", str(eff_id), style) if style else ("custom_cached", str(eff_id))
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
            Tuple[str, str], List[Tuple[Dict[str, Any], Dict[str, Any]]]
        ] = defaultdict(list)
        for item in self.queue_tts:
            if not item.get("text", "").strip() or Path(item["filename"]).is_file():
                continue
            voice_key, options = self._inference_options(item)
            queue_by_voice[voice_key].append((item, options))

        try:
            completed = 0
            for items in queue_by_voice.values():
                queue = [item for item, _ in items]
                options = items[0][1]
                audios = self.engine.infer_batch(
                    [item["text"] for item in queue],
                    **options,
                )
                if len(audios) != len(queue):
                    raise RuntimeError(
                        "VieNeu-TTS returned an unexpected number of audio segments"
                    )
                for item, audio in zip(queue, audios):
                    sf.write(item["filename"], audio, self.engine.sample_rate)
                    completed += 1
                    self.signal(text=f"TTS[{completed}/{self.len}]")
            self.signal(text="TTS ended")
        finally:
            self.engine.close()
