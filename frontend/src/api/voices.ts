import { apiRequest } from './client';

export interface CustomVoice {
  id: string;
  name: string;
  description?: string;
  provider: number;
  kind?: string;
  language?: string;
  ref_audio_path?: string;
  ref_text?: string;
  instruct?: string;
  external_voice_id?: string;
  tuning_params?: Record<string, any>;
  preview_audio_path?: string;
  is_active?: boolean;
  created_at?: number;
  updated_at?: number;
}

export interface CreateVoicePayload {
  name: string;
  provider?: number | string;
  description?: string;
  language?: string;
  kind?: string;
  ref_text?: string;
  instruct?: string;
  audio?: Blob | File;
  external_voice_id?: string;
  tuning_params?: Record<string, any>;
  denoise?: boolean;
  cut_start?: number;
  cut_end?: number;
  preview_filename?: string;
  preview_audio_path?: string;
}

export interface PreviewClonePayload {
  audio: Blob | File;
  provider?: number | string;
  text?: string;
  language?: string;
  denoise?: boolean;
  cut_start?: number;
  cut_end?: number;
  ref_text?: string;
}

export interface PreviewCloneResponse {
  ok: boolean;
  preview_id: string;
  preview_filename: string;
  preview_url: string;
  audio_url: string;
}

export interface TrimAudioPayload {
  audio: Blob | File;
  start_time?: number;
  end_time?: number;
  auto_detect?: boolean;
  target_duration?: number;
  provider?: number;
}

export interface TrimAudioResponse {
  ok: boolean;
  original_duration: number;
  start_time: number;
  end_time: number;
  duration: number;
  audio_url: string;
  filename: string;
}

export async function trimAudio(payload: TrimAudioPayload): Promise<TrimAudioResponse> {
  const formData = new FormData();
  let filename = 'audio.wav';
  if (payload.audio instanceof File) {
    filename = payload.audio.name;
  }
  formData.append('audio', payload.audio, filename);
  if (payload.start_time !== undefined) formData.append('start_time', String(payload.start_time));
  if (payload.end_time !== undefined) formData.append('end_time', String(payload.end_time));
  if (payload.auto_detect !== undefined) formData.append('auto_detect', String(payload.auto_detect));
  if (payload.target_duration !== undefined) formData.append('target_duration', String(payload.target_duration));
  if (payload.provider !== undefined) formData.append('provider', String(payload.provider));

  return apiRequest<TrimAudioResponse>('/api/voices/trim-audio', {
    method: 'POST',
    body: formData,
  });
}

export async function fetchCustomVoices(provider?: number): Promise<CustomVoice[]> {
  const url = provider !== undefined ? `/api/custom-voices?provider=${provider}` : '/api/custom-voices';
  const data = await apiRequest<{ voices: CustomVoice[] }>(url);
  return data.voices || [];
}

export async function fetchCustomVoice(id: string): Promise<CustomVoice> {
  return apiRequest<CustomVoice>(`/api/custom-voices/${encodeURIComponent(id)}`);
}

export async function createCustomVoice(payload: CreateVoicePayload): Promise<CustomVoice> {
  const formData = new FormData();
  formData.append('name', payload.name);
  if (payload.provider !== undefined) formData.append('provider', String(payload.provider));
  if (payload.description) formData.append('description', payload.description);
  if (payload.language) formData.append('language', payload.language);
  if (payload.kind) formData.append('kind', payload.kind);
  if (payload.ref_text) formData.append('ref_text', payload.ref_text);
  if (payload.instruct) formData.append('instruct', payload.instruct);
  if (payload.external_voice_id) formData.append('external_voice_id', payload.external_voice_id);
  if (payload.tuning_params) formData.append('tuning_params', JSON.stringify(payload.tuning_params));
  if (payload.denoise !== undefined) formData.append('denoise', String(payload.denoise));
  if (payload.cut_start !== undefined) formData.append('cut_start', String(payload.cut_start));
  if (payload.cut_end !== undefined) formData.append('cut_end', String(payload.cut_end));
  if (payload.preview_filename) formData.append('preview_filename', payload.preview_filename);
  if (payload.preview_audio_path) formData.append('preview_audio_path', payload.preview_audio_path);
  if (payload.audio) {
    let filename = 'recording.wav';
    if (payload.audio instanceof File) {
      filename = payload.audio.name;
    } else if (payload.audio.type) {
      if (payload.audio.type.includes('mpeg') || payload.audio.type.includes('mp3')) filename = 'recording.mp3';
      else if (payload.audio.type.includes('webm')) filename = 'recording.webm';
      else if (payload.audio.type.includes('mp4') || payload.audio.type.includes('m4a')) filename = 'recording.mp4';
      else if (payload.audio.type.includes('ogg')) filename = 'recording.ogg';
      else if (payload.audio.type.includes('flac')) filename = 'recording.flac';
    }
    formData.append('audio', payload.audio, filename);
  }

  return apiRequest<CustomVoice>('/api/custom-voices', {
    method: 'POST',
    body: formData,
  });
}

export async function previewCloneVoice(payload: PreviewClonePayload): Promise<PreviewCloneResponse> {
  const formData = new FormData();
  let filename = 'recording.wav';
  if (payload.audio instanceof File) {
    filename = payload.audio.name;
  } else if (payload.audio.type) {
    if (payload.audio.type.includes('mpeg') || payload.audio.type.includes('mp3')) filename = 'recording.mp3';
    else if (payload.audio.type.includes('webm')) filename = 'recording.webm';
    else if (payload.audio.type.includes('mp4') || payload.audio.type.includes('m4a')) filename = 'recording.mp4';
    else if (payload.audio.type.includes('ogg')) filename = 'recording.ogg';
    else if (payload.audio.type.includes('flac')) filename = 'recording.flac';
  }
  formData.append('audio', payload.audio, filename);
  if (payload.provider !== undefined) formData.append('provider', String(payload.provider));
  if (payload.text) formData.append('text', payload.text);
  if (payload.language) formData.append('language', payload.language);
  if (payload.denoise !== undefined) formData.append('denoise', String(payload.denoise));
  if (payload.cut_start !== undefined) formData.append('cut_start', String(payload.cut_start));
  if (payload.cut_end !== undefined) formData.append('cut_end', String(payload.cut_end));
  if (payload.ref_text) formData.append('ref_text', payload.ref_text);

  return apiRequest<PreviewCloneResponse>('/api/voices/preview-clone', {
    method: 'POST',
    body: formData,
  });
}

export async function updateCustomVoice(id: string, updates: Partial<CustomVoice>): Promise<CustomVoice> {
  return apiRequest<CustomVoice>(`/api/custom-voices/${encodeURIComponent(id)}`, {
    method: 'PUT',
    body: JSON.stringify(updates),
  });
}

export async function deleteCustomVoice(id: string, hard: boolean = false): Promise<{ ok: boolean; id: string }> {
  return apiRequest<{ ok: boolean; id: string }>(`/api/custom-voices/${encodeURIComponent(id)}${hard ? '?hard=true' : ''}`, {
    method: 'DELETE',
  });
}

export async function bulkDeleteCustomVoices(ids: string[], hard: boolean = false): Promise<{ ok: boolean; deleted: string[]; count: number }> {
  return apiRequest<{ ok: boolean; deleted: string[]; count: number }>('/api/custom-voices/bulk-delete', {
    method: 'POST',
    body: JSON.stringify({ ids, hard }),
  });
}

export async function previewCustomVoice(id: string, text?: string, language?: string): Promise<{ ok: boolean; preview_url: string }> {
  return apiRequest<{ ok: boolean; preview_url: string }>(`/api/custom-voices/${encodeURIComponent(id)}/preview`, {
    method: 'POST',
    body: JSON.stringify({ text, language }),
  });
}

export function getCustomVoiceAudioUrl(id: string): string {
  return `/api/custom-voices/${encodeURIComponent(id)}/audio`;
}

export function getCustomVoicePreviewAudioUrl(id: string): string {
  return `/api/custom-voices/${encodeURIComponent(id)}/preview/audio`;
}

export interface VoiceDesignAssistRequest {
  prompt: string;
  base_voice?: string;
}

export interface VoiceDesignAssistResponse {
  style: string;
  temperature: number;
  repetition_penalty: number;
  suggested_tags: string[];
  suggested_name: string;
  description: string;
}

export async function requestVoiceDesignAssist(payload: VoiceDesignAssistRequest): Promise<VoiceDesignAssistResponse> {
  return apiRequest<VoiceDesignAssistResponse>('/api/voices/design-assist', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export interface TTSPreviewPayload {
  text?: string;
  voice?: string;
  voice_id?: string;
  voiceId?: string;
  provider?: number | string;
  ttsType?: number | string;
  language?: string;
  speed?: number;
  rate?: string;
  pitch?: string;
  style?: string;
  tuning_params?: Record<string, any>;
  tuningParams?: Record<string, any>;
  segment_id?: number | string;
  segmentId?: number | string;
  force_refresh?: boolean;
  auto_speed?: boolean;
  autoSpeed?: boolean;
  slot_duration_s?: number;
  slotDuration?: number;
  max_speed_rate?: number;
  maxSpeedRate?: number;
}

export interface TTSPreviewResponse {
  ok: boolean;
  id: string;
  preview_id: string;
  preview_url: string;
  audio_url: string;
  voice?: string;
  provider?: number;
  applied_speed?: number;
}

export async function previewTTS(
  payload: TTSPreviewPayload,
  options?: RequestInit
): Promise<TTSPreviewResponse> {
  return apiRequest<TTSPreviewResponse>('/api/tts/preview', {
    method: 'POST',
    body: JSON.stringify(payload),
    ...options,
  });
}

export function getTTSPreviewAudioUrl(id: string): string {
  return `/api/tts/preview/${encodeURIComponent(id)}/audio`;
}

export interface BatchTTSSegmentPayload {
  segmentId: number | string;
  text: string;
  voice: string;
  provider: number | string;
  language?: string;
  speed?: number;
  style?: string;
  tuningParams?: Record<string, any>;
  forceRefresh?: boolean;
  autoSpeed?: boolean;
  slotDuration?: number;
  maxSpeedRate?: number;
}

export interface BatchTTSPayload {
  batchId?: string;
  segments: BatchTTSSegmentPayload[];
  language?: string;
  speed?: number;
  autoSpeed?: boolean;
  maxSpeedRate?: number;
}

export interface BatchTTSResultItem {
  segmentId: number | string;
  ok: boolean;
  id?: string | null;
  preview_id?: string | null;
  previewUrl?: string | null;
  preview_url?: string | null;
  audio_url?: string | null;
  durationSec?: number;
  appliedSpeed?: number;
  applied_speed?: number;
  voice?: string | null;
  provider?: number | null;
  error?: string | null;
}

export interface BatchTTSResponse {
  ok: boolean;
  batch_id?: string;
  succeeded?: number;
  failed?: number;
  results: BatchTTSResultItem[];
}

export async function batchTTS(
  payload: BatchTTSPayload,
  options?: RequestInit
): Promise<BatchTTSResponse> {
  const headers = new Headers(options?.headers || {});
  if (!headers.has('Accept')) headers.set('Accept', 'application/json');
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');

  const response = await fetch('/api/tts/batch', {
    method: 'POST',
    body: JSON.stringify(payload),
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorMsg = `Request failed: ${response.status} ${response.statusText}`;
    try {
      const text = await response.text();
      if (text && text.trim()) {
        try {
          const errJson = JSON.parse(text);
          if (errJson.detail) errorMsg = errJson.detail;
          else if (errJson.message) errorMsg = errJson.message;
          else if (errJson.error) errorMsg = errJson.error;
          else errorMsg = text.trim();
        } catch (_) {
          errorMsg = text.trim();
        }
      }
    } catch (_) {}
    const err = new Error(errorMsg) as Error & { status?: number };
    err.status = response.status;
    throw err;
  }

  return response.json();
}

export async function cancelBatchTTS(batchId: string): Promise<{ ok: boolean; batch_id: string; cancelled: boolean }> {
  return apiRequest<{ ok: boolean; batch_id: string; cancelled: boolean }>(
    `/api/tts/batch/${encodeURIComponent(batchId)}/cancel`,
    { method: 'POST' }
  );
}
