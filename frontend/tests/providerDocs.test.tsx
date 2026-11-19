import { describe, it, expect } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { PROVIDER_DOCS, getProviderDocInfo } from '../src/config/providerDocs';
import { ProviderDocLink } from '../src/components/ProviderDocLink';

describe('Provider Documentation Registry & Links', () => {
  it('resolves official documentation links for ASR providers', () => {
    const qwenDoc = getProviderDocInfo(0, 'asr');
    expect(qwenDoc).not.toBeNull();
    expect(qwenDoc?.id).toBe('qwen_asr');
    expect(qwenDoc?.docUrl).toContain('Qwen3-ASR');

    const deepgramDoc = getProviderDocInfo('deepgram', 'asr');
    expect(deepgramDoc).not.toBeNull();
    expect(deepgramDoc?.docUrl).toContain('developers.deepgram.com');

    const deepgramNumDoc = getProviderDocInfo(1, 'asr');
    expect(deepgramNumDoc?.id).toBe('deepgram');

    const geminiSttDoc = getProviderDocInfo(2, 'asr');
    expect(geminiSttDoc?.id).toBe('gemini_stt');
    expect(geminiSttDoc?.category).toBe('asr');

    const googleSttDoc = getProviderDocInfo(3, 'asr');
    expect(googleSttDoc?.id).toBe('google_stt');
    expect(googleSttDoc?.category).toBe('asr');

    const elevenSttDoc = getProviderDocInfo(4, 'asr');
    expect(elevenSttDoc?.id).toBe('elevenlabs_stt');
    expect(elevenSttDoc?.category).toBe('asr');

    const whisperDoc = getProviderDocInfo(5, 'asr');
    expect(whisperDoc).not.toBeNull();
    expect(whisperDoc?.id).toBe('whisper');
    expect(whisperDoc?.docUrl).toContain('platform.openai.com/docs/guides/speech-to-text');

    expect(getProviderDocInfo(99, 'asr')).toBeNull();

    const funasrDoc = getProviderDocInfo('funasr');
    expect(funasrDoc).not.toBeNull();
    expect(funasrDoc?.docUrl).toContain('github.com/modelscope/FunASR');
  });

  it('resolves official documentation links for LLM Translation providers', () => {
    const openaiDoc = getProviderDocInfo('openai', 'llm');
    expect(openaiDoc?.docUrl).toContain('platform.openai.com');

    const geminiDoc = getProviderDocInfo('gemini', 'llm');
    expect(geminiDoc?.docUrl).toContain('ai.google.dev');

    const deepseekDoc = getProviderDocInfo('deepseek', 'llm');
    expect(deepseekDoc?.docUrl).toContain('api-docs.deepseek.com');

    const claudeDoc = getProviderDocInfo('claude', 'llm');
    expect(claudeDoc?.docUrl).toContain('docs.anthropic.com');

    const googleDoc = getProviderDocInfo('google', 'llm');
    expect(googleDoc?.docUrl).toContain('cloud.google.com/translate/docs');
  });

  it('resolves official documentation links for TTS providers by number and name', () => {
    // VieNeu-TTS (id 2)
    const vieneuDoc = getProviderDocInfo(2, 'tts');
    expect(vieneuDoc?.docUrl).toContain('github.com/pnn-tech/vieneu-tts');

    // OmniVoice (id 1)
    const omniDoc = getProviderDocInfo(1, 'tts');
    expect(omniDoc?.docUrl).toContain('github.com/k2-fsa/OmniVoice');

    // ElevenLabs (id 0)
    const elevenDoc = getProviderDocInfo(0, 'tts');
    expect(elevenDoc?.docUrl).toContain('elevenlabs.io');

    // Edge-TTS (id 4)
    const edgeDoc = getProviderDocInfo(4, 'tts');
    expect(edgeDoc?.docUrl).toContain('github.com/rany2/edge-tts');

    // Shared name 'openai' with tts categoryHint
    expect(getProviderDocInfo('openai', 'tts')?.id).toBe('openaitts');
  });

  it('renders ProviderDocLink with secure new-tab attributes', () => {
    const html = renderToStaticMarkup(
      <ProviderDocLink providerId="deepgram" category="asr" variant="pill" />
    );
    expect(html).toContain('href="https://developers.deepgram.com');
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener noreferrer"');
    expect(html).toContain('Guide');
  });

  it('returns empty markup when provider is unknown', () => {
    const html = renderToStaticMarkup(
      <ProviderDocLink providerId="unknown_nonexistent_provider_xyz" />
    );
    expect(html).toBe('');
  });
});
