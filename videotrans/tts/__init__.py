from typing import Type, Union

from videotrans import ChannelProvider, get_class
from videotrans.configure.config import app_cfg, params, tr
from videotrans.tts._base import BaseTTS


ELEVENLABS_TTS = 0
OMNIVOICE_TTS = 1
VIENEU_TTS = 2
GEMINI_TTS = 3
DEFAULT_TTS = VIENEU_TTS

SUPPORT_CLONE = [OMNIVOICE_TTS, VIENEU_TTS]
LOCAL_BUILTIN = [OMNIVOICE_TTS]
CLOUD_TTS_PROVIDERS = frozenset({ELEVENLABS_TTS, GEMINI_TTS})
CHANGE_BY_LANGUAGE = []

_ID_NAME_DICT = {
    ELEVENLABS_TTS: ChannelProvider(
        "ElevenLabs", "._elevenlabs", key_name="elevenlabstts_key"
    ),
    OMNIVOICE_TTS: ChannelProvider(
        f"OmniVoice({tr('Built-in')})", "._omnivoice"
    ),
    VIENEU_TTS: ChannelProvider("VieNeu-TTS", "._vieneutts"),
    GEMINI_TTS: ChannelProvider(
        "Gemini TTS", "._geminitts", key_name="gemini_key"
    ),
}
TTS_NAME_LIST = [provider.name for provider in _ID_NAME_DICT.values()]


def is_allow_lang(langcode: str = None, tts_type: int = None):
    if langcode is None or tts_type is None:
        return True
    provider = _ID_NAME_DICT.get(tts_type)
    if provider is None:
        return f"Unsupported TTS provider: {tts_type}"
    if tts_type == VIENEU_TTS and langcode[:2] not in ["vi", "en"]:
        return provider.name + tr("Dubbing channel") + " " + tr("may not support") + tr(langcode)
    return True


def is_input_api(tts_type: int = None, return_str=False):
    provider = _ID_NAME_DICT.get(tts_type)
    if not provider:
        return True
    if provider.key_name and not params.get(provider.key_name):
        return "Please configure the SK or API information of the channel first."
    return True


def clone_tips(tts_type, role: str = "No", recogn_type=9):
    if tts_type in SUPPORT_CLONE and role == "clone":
        return tr("clone_dubb_tips1") + (tr("clone_dubb_tips2") if recogn_type < 2 else "")
    return None


def run(
    *,
    queue_tts=None,
    language=None,
    uuid=None,
    play=False,
    is_test=False,
    tts_type=DEFAULT_TTS,
    is_cuda=False,
    is_redubb=False,
    event_sink=None,
    cancellation_token=None,
) -> None:
    if len(queue_tts) < 1 or app_cfg.exit_soft or (uuid and uuid in app_cfg.stoped_uuid_set):
        return

    kwargs = {
        "queue_tts": queue_tts,
        "language": language,
        "uuid": uuid,
        "play": play,
        "is_test": is_test,
        "tts_type": tts_type,
        "is_cuda": is_cuda,
        "is_redubb": is_redubb,
        "event_sink": event_sink,
        "cancellation_token": cancellation_token,
    }
    provider_class: Union[Type[BaseTTS], None] = get_class(
        tts_type, "tts", _ID_NAME_DICT
    )
    if not provider_class:
        from videotrans.configure.excepts import DubbingSrtError

        raise DubbingSrtError(f"No this TTS Channel:{tts_type=}")
    return provider_class(**kwargs).run()  # type: ignore
