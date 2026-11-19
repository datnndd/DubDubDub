# -*- coding: utf-8 -*-
from dataclasses import dataclass
from videotrans.configure.config import params
from videotrans.translator._openaicompat import OpenAICampat

@dataclass
class DeepSeek(OpenAICampat):

    def __post_init__(self):
        self.ainame="deepseek"
        raw_model = (params.get('deepseek_model', "deepseek-chat") or "deepseek-chat").strip()
        if not raw_model or raw_model.startswith("deepseek-v4"):
            raw_model = "deepseek-chat"
        self.model_name = raw_model
        self.api_url = params.get('deepseek_api', 'https://api.deepseek.com/v1')
        self.api_key = params.get('deepseek_key', '')
        try:
            raw_max = int(params.get('deepseek_max_token', 8192) or 8192)
        except (TypeError, ValueError):
            raw_max = 8192
        thinking_enabled = bool(params.get('deepseek_thinking')) or "reasoner" in self.model_name.lower()
        max_cap = 32768 if thinking_enabled else 8192
        self.max_tokens = max(1, min(raw_max, max_cap))
        self.reasoning_effort="high" if params.get('deepseek_thinking') else None
        self.extra_body={"thinking": {"type": "enabled" if params.get('deepseek_thinking') else "disabled"}}
        super().__post_init__()
