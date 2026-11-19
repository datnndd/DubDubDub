# -*- coding: utf-8 -*-
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from videotrans.configure._paths import ROOT_DIR
from videotrans.configure._logging import _write_with_retry
from videotrans.configure.contants import (
    DEFAULT_GEMINI_MODEL,
    ELEVENLABS_TTS_MODELS,
    GEMINI_TTS_MODELS,
    Chatgpt_Model,
    Deepseek_Model,
)

# Module-level reference to settings singleton, set by config.py
_settings_ref = None

PROVIDER_CATALOG_VERSION = 3
_LEGACY_ASR_IDS = {0: 5, 2: 0, 15: 2, 20: 4, 21: 3, 22: 1}
_LEGACY_TRANSLATION_IDS = {0: 0, 4: 1, 5: 3, 6: 2}
_LEGACY_TTS_IDS = {3: 1, 21: 3, 22: 0, 34: 2}
_DEFAULT_TTS_ID = 2
_DEFAULT_ASR_MODELS = {
    0: "1.7B",
    1: "nova-3",
    2: "gemini-flash-latest",
    3: "google-web-speech",
    4: "scribe_v2",
    5: "large-v3",
}


def set_settings_ref(settings):
    global _settings_ref
    _settings_ref = settings


@dataclass
class AppParams:
    """
    AppParams: corresponds to params.json, contains getset_params functionality
    """
    _json_path: str = f"{ROOT_DIR}/videotrans/params.json"

    def __post_init__(self):
        self.getset_params()

    def save(self, data: Dict = None):
        if data:
            return self.getset_params(data)
        self._save_to_disk()

    def getset_params(self, update_data: Dict = None) -> Dict:
        if update_data:
            self._apply_dict(update_data)
            self._save_to_disk()
            return self.to_dict()

        default = self._get_defaults()
        migrated = False
        if Path(self._json_path).exists():
            try:
                loaded = json.loads(Path(self._json_path).read_text(encoding='utf-8'))
                catalog_version = int(loaded.get("provider_catalog_version", 1))
                if catalog_version < 2:
                    for key in ("recogn_type", "stt_recogn_type"):
                        loaded[key] = _LEGACY_ASR_IDS.get(int(loaded.get(key, 22)), 1)
                    loaded["model_name"] = _DEFAULT_ASR_MODELS[loaded["recogn_type"]]
                    loaded["stt_model_name"] = _DEFAULT_ASR_MODELS[loaded["stt_recogn_type"]]
                    for key in ("translate_type", "trans_translate_type"):
                        loaded[key] = _LEGACY_TRANSLATION_IDS.get(int(loaded.get(key, 0)), 0)
                    migrated = True
                if catalog_version < 3:
                    for key in ("tts_type", "dubb_tts_type"):
                        loaded[key] = _LEGACY_TTS_IDS.get(
                            int(loaded.get(key, 34)), _DEFAULT_TTS_ID
                        )
                    loaded["provider_catalog_version"] = PROVIDER_CATALOG_VERSION
                    migrated = True
                # Update f5tts_role separately
                loaded['f5tts_role']=("\n".join(set( (str(loaded.get('f5tts_role', '')).strip()+"\n"+default['f5tts_role']).split("\n") ) )).strip()
                filtered = {key: value for key, value in loaded.items() if key in default}
                if len(filtered) != len(loaded):
                    migrated = True
                default.update(filtered)
            except (OSError, json.JSONDecodeError):
                pass
        else:
            self._apply_dict(default)
            self._save_to_disk()

        self._apply_dict(default)
        if migrated:
            self._save_to_disk()
        return self.to_dict()

    def _get_defaults(self):
        _settings = _settings_ref
        return {
            "provider_catalog_version": PROVIDER_CATALOG_VERSION,
            "last_opendir": os.path.expanduser("~"),
            "output_dir": "",
            "is_cuda": False,
            "line_roles": {},
            "rephrase": 0,
            "is_separate": False,
            "clear_cache": True,
            "embed_bgm": True,
            "remove_noise": False,
            "enable_diariz": False,
            "nums_diariz": 0,
            "target_dir": "",
            "source_language": "en",
            "target_language": "zh-cn",
            "translate_type": 0,
            "subtitle_type": 2,
            "tts_type": _DEFAULT_TTS_ID,
            "model_name": "nova-3",
            "recogn_type": 1,
            "fix_punc": 0,
            "stt_fix_punc": 0,
            "voice_autorate": True,
            "video_autorate": False,
            "align_sub_audio": True,
            "voice_role": "No",
            "voice_rate": "0",
            "chatgpt_api": "",
            "chatgpt_key": "",
            "chatgpt_reasoning_effort": "No",
            "chatgpt_max_token": 16384,
            "chatgpt_model": (lambda m: Chatgpt_Model.split(',')[0] if not m or m == '-' or m.startswith(('gpt-5.5', 'gpt-5.4', 'gpt-5')) else m)(str(_settings.get('chatgpt_model', '') or '').strip().split(',')[0]),
            "gemini_key": "",
            "gemini_api": "",
            "gemini_model": (lambda m: DEFAULT_GEMINI_MODEL.split(',')[0] if not m or m == '-' or m.startswith(('gemini-3.6', 'gemini-3.5')) else m)(str(_settings.get('gemini_model', '') or '').strip().split(',')[0]),
            "gemini_maxtoken": 16384,
            "gemini_ttsstyle": "",
            "gemini_ttsmodel": (lambda m: GEMINI_TTS_MODELS.split(',')[0] if not m or m == '-' or m.startswith('gemini-3.1') else m)(str(_settings.get('gemini_ttsmodel', '') or '').strip().split(',')[0]),
            "deepseek_key": "",
            "deepseek_api": "https://api.deepseek.com/v1",
            "deepseek_model": (lambda m: Deepseek_Model.split(',')[0] if not m or m == '-' or m.startswith('deepseek-v4') else m)(str(_settings.get('deepseek_model', '') or '').strip().split(',')[0]),
            "deepseek_max_token": 8192,
            "elevenlabstts_role": [],
            "elevenlabstts_key": "",
            "elevenlabstts_models": (lambda m: ELEVENLABS_TTS_MODELS.split(',')[0] if not m or m == '-' or m.startswith('eleven_v3') else m)(str(_settings.get('elevenlabstts_models', '') or '').strip().split(',')[0]),
            "omnivoice_url": "http://127.0.0.1:7860",
            "f5tts_role": "nverguo.wav#你说四大皆空，却为何，紧闭双眼，若你睁开眼睛看看我，我不相信你，两眼空空。\ncosy.wav#希望你以后，能够做的比我还好哟！\nzh_male_bj.wav#说起咱北京的烤鸭啊，那可真是外焦里嫩、色泽金黄，一口咬下去满嘴流油！\nzh_female_cn.wav#大家好呀，今天跟你们分享一下我的日常护肤小习惯，其实护肤不需要太复杂，清洁补水最重要。\nzh_female_tw.wav#台湾有许多隐藏版的小吃店，他们可能不起眼，但食物却十分美味，下次不妨多留意身边这样的小店吧！",
            "vieneu_roles": {},
            "app_mode": "biaozhun",
            "stt_source_language": 0,
            "stt_recogn_type": 1,
            "stt_model_name": "nova-3",
            "stt_cuda": False,
            "stt_remove_noise": False,
            "stt_enable_diariz": False,
            "stt_rephrase": 0,
            "stt_nums_diariz": 0,
            "subtitlecover_outformat": "srt",
            "deepgram_apikey": "",
            "trans_translate_type": 0,
            "trans_source_language": 0,
            "trans_target_language": 1,
            "trans_out_format": 0,
            "dubb_source_language": 0,
            "dubb_tts_type": _DEFAULT_TTS_ID,
            "dubb_role": 0,
            "dubb_out_format": 0,
            "dubb_voice_autorate": False,
            "dubb_hecheng_rate": 0,
            "dubb_pitch_rate": 0,
            "dubb_volume_rate": 0,
            "recogn2pass": False
        }

    def _apply_dict(self, data: Dict):
        for k, v in data.items():
            if k in self._secret_keys():
                if v:
                    try:
                        from videotrans.core.secret_store import set_secret
                        set_secret(k, str(v).strip())
                    except Exception:
                        pass
                setattr(self, k, "")
            else:
                setattr(self, k, v)

    def _secret_keys(self) -> set[str]:
        try:
            from videotrans.core.secret_store import SECRET_KEYS
            return SECRET_KEYS
        except Exception:
            return {"deepgram_apikey", "chatgpt_key", "gemini_key", "deepseek_key", "elevenlabstts_key", "hf_token"}

    def to_dict(self) -> Dict:
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        for sk in self._secret_keys():
            if sk in data:
                data[sk] = ""
        return data

    def _save_to_disk(self):
        try:
            data = self.to_dict()
            _write_with_retry(self._json_path, json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logging.getLogger('VideoTrans').exception(f'Failed to save params locally: {e}', exc_info=True)

    def __getattribute__(self, item):
        if not item.startswith('_') and item in {
            "deepgram_apikey", "chatgpt_key", "gemini_key", "deepseek_key", "elevenlabstts_key", "hf_token"
        }:
            try:
                from videotrans.core.secret_store import resolve_secret
                val = resolve_secret(item)
                if val is not None and val.strip():
                    return val
            except Exception:
                pass
            return ""
        return super().__getattribute__(item)

    def __getitem__(self, item):
        if item in self._secret_keys():
            return self.get(item, "")
        return getattr(self, item)

    def __setitem__(self, key, value):
        if key in self._secret_keys():
            try:
                from videotrans.core.secret_store import set_secret
                set_secret(key, str(value or "").strip())
            except Exception:
                pass
            setattr(self, key, "")
            return
        setattr(self, key, value)

    def get(self, key, default=None):
        if key in self._secret_keys():
            try:
                from videotrans.core.secret_store import resolve_secret
                resolved = resolve_secret(key)
                if resolved is not None and resolved.strip():
                    return resolved
            except Exception:
                pass
            return default if default is not None else ""
        return getattr(self, key, default)

