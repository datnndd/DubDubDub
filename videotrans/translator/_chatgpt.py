from videotrans.configure.config import  params
from videotrans.translator._openaicompat import OpenAICampat
from dataclasses import dataclass

from videotrans.configure.config import params
from videotrans.translator._openaicompat import OpenAICampat


@dataclass
class ChatGPT(OpenAICampat):
    def __post_init__(self):
        self.ainame = "chatgpt"
        self.api_key = params.get('chatgpt_key', '')
        self.api_url = params.get('chatgpt_api', '')
        self.max_tokens = int(params.get('chatgpt_max_token', 8192))
        raw_model = (params.get("chatgpt_model", "") or "gpt-4o").strip()
        if not raw_model or raw_model.startswith(("gpt-5.5", "gpt-5.4", "gpt-5")):
            raw_model = "gpt-4o"
        self.model_name = raw_model
        _reason=params.get('chatgpt_reasoning_effort')
        self.reasoning_effort=None if not _reason or _reason=='No' else _reason
        super().__post_init__()



