export interface ProjectMetadata {
  filename: string;
  format: string;
  resolution: string;
  fps: string | number;
  duration: string;
  durationSec: number;
  fileSize: string;
  videoCodec: string;
  audioCodec: string;
  bitrate?: string;
  hasAudio: boolean;
  hasVideo: boolean;
  verified: boolean;
  lastSaved?: string;
}

export interface ProjectRecord {
  id: string;
  name: string;
  stage: number;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'paused';
  parent_id?: string | null;
  parentId?: string | null;
  media_path?: string;
  duration?: number;
  created_at?: string;
  updated_at?: string;
  state_json?: string;
}

export interface BackendOptionItem {
  id?: string | number;
  code?: string;
  name?: string;
  label?: string;
  models?: string[];
  recognType?: number;
  translateType?: number;
  ttsType?: number;
  thirdParty?: boolean;
  requiresSettings?: boolean;
  configured?: boolean;
  testable?: boolean;
}

export interface BackendOptions {
  languages: Array<{ code: string; name: string }>;
  asrProviders: BackendOptionItem[];
  translationProviders: BackendOptionItem[];
  translationModes: Array<{ id: string; label: string }>;
  voices: Array<{ id: string; name: string; lang?: string; gender?: string }>;
}
