/**
 * Centralized registry of official documentation links and guide URLs
 * for all ASR, LLM (Translation), and TTS providers supported in DubDub.
 */

export interface ProviderDocInfo {
  id: string;
  name: string;
  category: 'asr' | 'llm' | 'tts';
  docUrl: string;
  description: string;
}

export const PROVIDER_DOCS: Record<string, ProviderDocInfo> = {
  // --- ASR (Speech-to-Text) ---
  qwen_asr: {
    id: 'qwen_asr',
    name: 'Qwen3-ASR',
    category: 'asr',
    docUrl: 'https://github.com/QwenLM/Qwen3-ASR',
    description: 'Local multilingual speech recognition models powered by Qwen3-ASR.',
  },
  deepgram: {
    id: 'deepgram',
    name: 'Deepgram',
    category: 'asr',
    docUrl: 'https://developers.deepgram.com/docs/getting-started-with-pre-recorded-audio',
    description: 'Real-time & pre-recorded speech recognition API with speaker diarization and Nova-3 models.',
  },
  gemini_stt: {
    id: 'gemini_stt',
    name: 'Gemini STT',
    category: 'asr',
    docUrl: 'https://ai.google.dev/gemini-api/docs/audio',
    description: 'Multimodal audio transcription and timestamped speech recognition via Google Gemini.',
  },
  google_stt: {
    id: 'google_stt',
    name: 'Google Speech-to-Text',
    category: 'asr',
    docUrl: 'https://cloud.google.com/speech-to-text/docs',
    description: 'Google Speech-to-Text recognition API supporting global languages.',
  },
  elevenlabs_stt: {
    id: 'elevenlabs_stt',
    name: 'ElevenLabs Scribe',
    category: 'asr',
    docUrl: 'https://elevenlabs.io/docs/capabilities/speech-to-text',
    description: 'ElevenLabs Scribe speech-to-text API with high-accuracy transcription and diarization.',
  },
  whisper: {
    id: 'whisper',
    name: 'OpenAI Whisper',
    category: 'asr',
    docUrl: 'https://platform.openai.com/docs/guides/speech-to-text',
    description: 'OpenAI Whisper speech-to-text recognition API and open-source models.',
  },
  funasr: {
    id: 'funasr',
    name: 'FunASR',
    category: 'asr',
    docUrl: 'https://github.com/modelscope/FunASR',
    description: 'Fundamental end-to-end speech recognition toolkit by Alibaba ModelScope.',
  },

  // --- LLM & Translation ---
  google: {
    id: 'google',
    name: 'Google Translate',
    category: 'llm',
    docUrl: 'https://cloud.google.com/translate/docs',
    description: 'Neural Machine Translation API by Google Cloud.',
  },
  openai: {
    id: 'openai',
    name: 'OpenAI ChatGPT',
    category: 'llm',
    docUrl: 'https://platform.openai.com/docs/overview',
    description: 'Leading frontier LLMs including GPT-4o, GPT-4o-mini, and ChatGPT developer platform.',
  },
  gemini: {
    id: 'gemini',
    name: 'Google Gemini',
    category: 'llm',
    docUrl: 'https://ai.google.dev/gemini-api/docs',
    description: 'Multimodal AI models by Google DeepMind with long-context translation capabilities.',
  },
  deepseek: {
    id: 'deepseek',
    name: 'DeepSeek',
    category: 'llm',
    docUrl: 'https://api-docs.deepseek.com/',
    description: 'High-performance cost-effective reasoning & translation models by DeepSeek.',
  },
  claude: {
    id: 'claude',
    name: 'Anthropic Claude',
    category: 'llm',
    docUrl: 'https://docs.anthropic.com/en/docs/welcome',
    description: 'Claude 3.5 Sonnet & Haiku models with nuanced multilingual reasoning.',
  },
  ollama: {
    id: 'ollama',
    name: 'Ollama',
    category: 'llm',
    docUrl: 'https://ollama.com/library',
    description: 'Local self-hosted LLMs including Llama 3, Mistral, and Qwen.',
  },

  // --- TTS (Text-to-Speech) ---
  vieneu: {
    id: 'vieneu',
    name: 'VieNeu-TTS',
    category: 'tts',
    docUrl: 'https://github.com/pnn-tech/vieneu-tts',
    description: 'Vietnamese neural text-to-speech with natural prosody, voice design, and 0-shot cloning.',
  },
  omnivoice: {
    id: 'omnivoice',
    name: 'OmniVoice',
    category: 'tts',
    docUrl: 'https://github.com/k2-fsa/OmniVoice',
    description: 'Next-generation zero-shot voice cloning and cross-lingual TTS by next-gen Kaldi.',
  },
  elevenlabs: {
    id: 'elevenlabs',
    name: 'ElevenLabs',
    category: 'tts',
    docUrl: 'https://elevenlabs.io/docs/overview',
    description: 'Industry-standard ultra-realistic synthetic voices, emotion controls, and voice cloning.',
  },
  gemini_tts: {
    id: 'gemini_tts',
    name: 'Gemini TTS',
    category: 'tts',
    docUrl: 'https://ai.google.dev/gemini-api/docs/speech',
    description: 'Direct audio synthesis using Google Gemini speech and audio generation models.',
  },
  edgetts: {
    id: 'edgetts',
    name: 'Edge-TTS',
    category: 'tts',
    docUrl: 'https://github.com/rany2/edge-tts',
    description: 'Free Microsoft Edge cloud neural text-to-speech service with hundreds of global voices.',
  },
  openaitts: {
    id: 'openaitts',
    name: 'OpenAI TTS',
    category: 'tts',
    docUrl: 'https://platform.openai.com/docs/guides/text-to-speech',
    description: 'Natural speech generation with Alloy, Echo, Fable, Onyx, Nova, and Shimmer voices.',
  },
  cosyvoice: {
    id: 'cosyvoice',
    name: 'CosyVoice',
    category: 'tts',
    docUrl: 'https://github.com/FunAudioLLM/CosyVoice',
    description: 'Multi-lingual large voice generation and cross-lingual voice cloning by Alibaba.',
  },
  f5tts: {
    id: 'f5tts',
    name: 'F5-TTS',
    category: 'tts',
    docUrl: 'https://github.com/SWivid/F5-TTS',
    description: 'Fair non-autoregressive flow-matching speech synthesis with zero-shot voice cloning.',
  },
  chattts: {
    id: 'chattts',
    name: 'ChatTTS',
    category: 'tts',
    docUrl: 'https://github.com/2noise/ChatTTS',
    description: 'Conversational speech synthesis model optimized for dialogue, laughter, and pauses.',
  },
};

/**
 * Resolves a doc link info object by provider ID, enum code, or display name.
 */
export function getProviderDocInfo(
  idOrType: string | number | undefined | null,
  categoryHint?: 'asr' | 'llm' | 'tts'
): ProviderDocInfo | null {
  if (idOrType === undefined || idOrType === null) return null;
  const raw = String(idOrType).trim().toLowerCase();
  if (!raw) return null;

  const isNumeric = typeof idOrType === 'number' || /^\d+$/.test(raw);

  // Numerical mapping for ASR providers (matches videotrans/recognition/__init__.py)
  if (categoryHint === 'asr' && isNumeric) {
    const num = Number(raw);
    switch (num) {
      case 0: return PROVIDER_DOCS.qwen_asr;
      case 1: return PROVIDER_DOCS.deepgram;
      case 2: return PROVIDER_DOCS.gemini_stt;
      case 3: return PROVIDER_DOCS.google_stt;
      case 4: return PROVIDER_DOCS.elevenlabs_stt;
      case 5: return PROVIDER_DOCS.whisper;
      default: return null;
    }
  }

  // Numerical mapping for LLM translation providers (matches videotrans/translator/_registry.py)
  if (categoryHint === 'llm' && isNumeric) {
    const num = Number(raw);
    switch (num) {
      case 0: return PROVIDER_DOCS.google;
      case 1: return PROVIDER_DOCS.openai;
      case 2: return PROVIDER_DOCS.gemini;
      case 3: return PROVIDER_DOCS.deepseek;
      default: return null;
    }
  }

  // Numerical mapping for TTS providers
  if ((categoryHint === 'tts' || !categoryHint) && isNumeric) {
    const num = Number(raw);
    switch (num) {
      case 0: return PROVIDER_DOCS.elevenlabs;
      case 1: return PROVIDER_DOCS.omnivoice;
      case 2: return PROVIDER_DOCS.vieneu;
      case 3: return PROVIDER_DOCS.gemini_tts;
      case 4: return PROVIDER_DOCS.edgetts;
      case 5: return PROVIDER_DOCS.openaitts;
      case 6: return PROVIDER_DOCS.cosyvoice;
      case 7: return PROVIDER_DOCS.f5tts;
      case 8: return PROVIDER_DOCS.chattts;
      default: return null;
    }
  }

  // Category-specific string disambiguation before generic direct match
  if (raw === 'qwen-asr' || raw === 'qwen_asr' || raw === 'qwenasr' || raw.includes('qwen')) {
    return PROVIDER_DOCS.qwen_asr;
  }
  if (raw === 'whisper-large-v3' || raw === 'faster-whisper' || raw.includes('whisper')) {
    return PROVIDER_DOCS.whisper;
  }
  if (raw === 'gemini-stt' || (categoryHint === 'asr' && raw.includes('gemini'))) {
    return PROVIDER_DOCS.gemini_stt;
  }
  if (raw === 'google-stt' || (categoryHint === 'asr' && raw.includes('google'))) {
    return PROVIDER_DOCS.google_stt;
  }
  if (categoryHint === 'asr' && raw.includes('eleven')) {
    return PROVIDER_DOCS.elevenlabs_stt;
  }
  if (raw === 'gemini-tts' || raw === 'gemini_tts' || (categoryHint === 'tts' && raw.includes('gemini'))) {
    return PROVIDER_DOCS.gemini_tts;
  }

  // Direct ID match (respect categoryHint when specified)
  if (PROVIDER_DOCS[raw] && (!categoryHint || PROVIDER_DOCS[raw].category === categoryHint)) {
    return PROVIDER_DOCS[raw];
  }

  // Name or partial match
  if (raw.includes('deepgram')) return PROVIDER_DOCS.deepgram;
  if (raw.includes('funasr')) return PROVIDER_DOCS.funasr;
  if (raw.includes('vieneu')) return PROVIDER_DOCS.vieneu;
  if (raw.includes('omni')) return PROVIDER_DOCS.omnivoice;
  if (raw.includes('eleven')) return PROVIDER_DOCS.elevenlabs;
  if (raw.includes('gemini')) return PROVIDER_DOCS.gemini;
  if (raw.includes('deepseek')) return PROVIDER_DOCS.deepseek;
  if (raw.includes('claude') || raw.includes('anthropic')) return PROVIDER_DOCS.claude;
  if (raw.includes('ollama')) return PROVIDER_DOCS.ollama;
  if (raw.includes('edge')) return PROVIDER_DOCS.edgetts;
  if (raw.includes('openai') || raw.includes('chatgpt')) {
    return categoryHint === 'tts' ? PROVIDER_DOCS.openaitts : PROVIDER_DOCS.openai;
  }
  if (raw.includes('cosy')) return PROVIDER_DOCS.cosyvoice;
  if (raw.includes('f5')) return PROVIDER_DOCS.f5tts;
  if (raw.includes('chattts')) return PROVIDER_DOCS.chattts;
  if (raw.includes('google')) return PROVIDER_DOCS.google;

  return null;
}
